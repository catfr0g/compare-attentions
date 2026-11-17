# FLOPs DeltaNet: политика `algorithmic_matmul_conv_v2`

## Назначение и границы

Это версионированная **алгоритмическая метрика матричных произведений, свёрток и
свёрток индексов состояния**, а не число аппаратных инструкций и не оценка всего
floating-point compute. Один multiply-accumulate считается как 2 FLOPs. Для
произведения `(m,k) @ (k,n)` стоимость равна `2*m*k*n`, в том числе для матриц с
треугольной маской и нулевым начальным состоянием. Метрика не зависит от BF16/FP32,
seed, компиляции, autotune и выбора физических tile-размеров.

Включены dense-проекции q/k/v/beta/o, три depthwise short-conv ширины 4, операции
WY/chunk или recurrent-состояния, их производные и **явно выполняемая библиотекой
рекомпутация**. Outer-product обновления состояния включены, даже когда Triton
записывает их как elementwise multiply и add, а не `tl.dot`.

Исключены scalar elementwise-операции, SiLU/sigmoid, масштабирование beta/scale,
суммы для градиента beta, сложения вне MAC, q/k L2-нормализация и output RMSNorm,
включая их параметры и производные. Нулевые компоненты `qk_l2norm` и
`output_rmsnorm` означают **исключение политикой**, а не отсутствие работы.
Исключены также dropout, внешний GPT LayerNorm/MLP/residual, embedding,
loss/optimizer/validation и аппаратный padding по каналам. MLP и прочие dense
операции GPT учитываются отдельно общим estimator.

Отличайте исключённые scalar reductions (например, норма или `dbeta`) от
включённых state matvec и row-solve: это bilinear contractions алгоритма, хотя
в исходнике они также могут содержать `tl.sum`. Нельзя исключить всё, что имеет
синтаксис elementwise, иначе recurrent-состояние ошибочно исчезнет из счётчика.

Старые бюджеты `FlopCounterMode` не эквивалентны этой метрике. Сравнение двух
семейств допустимо только при одной явно записанной политике v2.

## Поддерживаемый контракт

`delta_attention_cost(batch_size, sequence_length, hidden_size, num_heads)`
возвращает компоненты `{name: {'forward': int, 'backward': int}}` **одного**
полностью обучаемого attention-адаптера. `backward` уже включает рекомпутацию;
повторно прибавлять forward нельзя. Функция не создаёт модель и не исполняет её.

Поддерживается только неизменённый default-конструктор FLA 0.6.0:

- `DeltaNet(d_model=D, num_heads=H, layer_idx=...)`;
- `expand_k=expand_v=1`, `use_beta=True`, `use_gate=False`;
- `mode='chunk'`, `use_short_conv=True`, `conv_size=4`, `conv_bias=False`;
- `qk_activation='silu'`, `qk_norm='l2'`, `allow_neg_eigval=False`;
- без attention mask, packed/variable-length sequences, cache, initial/final
  state и activation checkpointing; все float-входы/параметры требуют градиента;
- `B,T,D,H` — положительные обычные Python `int`, не bool/SymInt/float;
- `D` делится на `H`; `d=D/H` принадлежит `{16,32,64,128}`;
- при `1 <= T <= 64` реальный адаптер переключается на `fused_recurrent`;
- при `T > 64` требуется `T % 64 == 0`, размер chunk строго 64.

Неполный последний chunk отклоняется. Мы не подменяем его физический padding
фиктивным подсчётом реальных токенов. В convolution учитываются четыре логических
tap на каждый output, включая нулевое левое causal-padding; физические лишние
строки/каналы Triton не учитываются. Это согласуется с dense-square causal
convention классического внимания: masked коэффициенты не уменьшают размер GEMM.

Функция отклоняет версии `flash-linear-attention`/`fla-core`, отличные от 0.6.0,
не-Triton short-conv, включённые FLA TileLang/Gluon overrides и установленный
TileLang (он может включить альтернативный backend автоматически). Это
консервативное fail-closed поведение. По четырём размерам невозможно распознать
произвольно изменённый Python layer: вызывающий код обязан разрешать только
штатный адаптер/конфигурацию репозитория, а не передавать размеры неизвестного
варианта. Runtime-проверка не является проверкой целостности установленного wheel.

## Проверенные первичные исходники

Установлены `flash-linear-attention==0.6.0`, `fla-core==0.6.0`. Локальные исходники
побайтно сравнены с тегом `v0.6.0`, commit
`413ad2c655ece57b94bc9ddc51fdbd43cf09d984`. Это источники формул, а не предположение
о поведении из статьи:

1. [Адаптер DeltaNet, defaults/dispatch, строки 75–168, 186–188, 195–286](https://github.com/fla-org/flash-linear-attention/blob/413ad2c655ece57b94bc9ddc51fdbd43cf09d984/fla/layers/delta_net.py).
2. [Chunk forward/backward, строки 18–148](https://github.com/fla-org/flash-linear-attention/blob/413ad2c655ece57b94bc9ddc51fdbd43cf09d984/fla/ops/delta_rule/chunk.py).
3. [WY, recompute и производная inverse, строки 74–91, 150–200, 207–237](https://github.com/fla-org/flash-linear-attention/blob/413ad2c655ece57b94bc9ddc51fdbd43cf09d984/fla/ops/delta_rule/wy_fast.py).
4. [Block inverse64, строки 204–323](https://github.com/fla-org/flash-linear-attention/blob/413ad2c655ece57b94bc9ddc51fdbd43cf09d984/fla/ops/utils/solve_tril.py).
5. [Chunk state forward/reverse, строки 211–316, 540–651](https://github.com/fla-org/flash-linear-attention/blob/413ad2c655ece57b94bc9ddc51fdbd43cf09d984/fla/ops/common/chunk_delta_h.py).
6. [Chunk output, local dv и q/k/w VJP](https://github.com/fla-org/flash-linear-attention/blob/413ad2c655ece57b94bc9ddc51fdbd43cf09d984/fla/ops/common/chunk_o.py).
7. [Recurrent: forward 76–91, backward 172–248, сохранение innovation 415](https://github.com/fla-org/flash-linear-attention/blob/413ad2c655ece57b94bc9ddc51fdbd43cf09d984/fla/ops/delta_rule/fused_recurrent.py).
8. [Short-conv: preactivation recompute в backward, строки 743–756; default backend 1140–1174](https://github.com/fla-org/flash-linear-attention/blob/413ad2c655ece57b94bc9ddc51fdbd43cf09d984/fla/modules/causal_conv1d/ops.py).

## Вывод chunk-формул

Обозначения: `C=64`, `K=V=d=D/H`, `N=B*H*(T/C)`; таблица ниже даёт стоимость
**на один chunk одной головы**. Множитель `N` применяется только после вывода.
Все матричные маски накладываются после dense contraction.

Для chunk зададим `L=tril(beta*K*K^T,-1)`, `A=(I+L)^-1`,
`W=A*(beta*K)`, `U=A*(beta*V)`. Здесь звёздочка в `A*(...)` означает GEMM;
масштабирование beta остаётся elementwise и исключено. Состояние и выход:

```text
Vnew = U - W @ S
Snext = S + K.T @ Vnew
O = (Q @ S + tril(Q @ K.T) @ Vnew) * scale
```

| Компонент | Forward | Backward |
|---|---:|---:|
| `chunk_gram` | `2*C²*d` | 0 |
| `chunk_inverse` | `4*14*(2*16²) + 16*(2*16³) = 159744` | 0 |
| `chunk_wy` | `4*C²*d` | 0 |
| `chunk_state` | `4*C*d²` | 0 |
| `chunk_output` | `2*C*d² + 4*C²*d` | 0 |
| `chunk_recompute_wy` | 0 | `4*C²*d` |
| `chunk_recompute_state` | 0 | `4*C*d²` |
| `chunk_local_dv` | 0 | `4*C²*d` |
| `chunk_reverse_state` | 0 | `6*C*d²` |
| `chunk_qkw_grad` | 0 | `6*C*d² + 6*C²*d` |
| `chunk_wy_grad` | 0 | `12*C²*d + 4*C³` |

### Почему inverse не равен условному `2*C³/3`

FLA решает четыре diagonal блока 16×16. Каждый выполняет row-solve для строк
2…15: 14 произведений `(1,16) @ (16,16)`. Затем вычисляются шесть off-diagonal
inverse-блоков. Для первой subdiagonal нужны `3*2` GEMM 16³, для следующей `2*3`,
для последней `1*4`: **16**, а не 15 GEMM. Scalar знаки, identity и промежуточные
сложения исключены. Исполняемый shape-ledger выявил ошибку ручного коэффициента
15 до завершения реализации.

### Почему backward не равен двум forward

`chunk_delta_rule_bwd` заново вычисляет W/U, S и Vnew. Он не пересчитывает Gram и
block inverse, потому что сохраняет A. Затем выполняются:

1. Local `dVnew`: `tril(QK.T).T @ dO`, два GEMM.
2. Reverse state: `K @ dSnext`, `Q.T @ dO`, `W.T @ dU`, три GEMM.
3. Q/K/W VJP: `dO @ Vnew.T`, `dO @ S.T`, `Vnew @ dSnext.T`,
   `dU @ S.T`, `dScores @ K`, `dScores.T @ Q`.
4. WY VJP: четыре GEMM размеров с C²d, два GEMM C³ для
   `dL = tril(-A.T @ tril(dA,-1) @ A.T,-1)`, ещё два GEMM C²d для двух
   вкладов от K в beta*K*K.T. Итого `12*C²*d + 4*C³`.

Таким образом, core-forward равен `N*(6*C*d² + 10*C²*d + 159744)`,
core-backward — `N*(16*C*d² + 26*C²*d + 4*C³)`.
Рекомпутация уже входит в последнее выражение.

## Short sequence и остальные компоненты

Для `T<=64` forward выполняет три state contraction/token:
`k.T @ S`, outer update `k @ innovation`, `q.T @ Snext`.
Каждая стоит `2*d²`; core-forward = `B*H*T*6*d²`.

Custom backward сохраняет innovation до умножения на beta. Reverse-проход
выполняет четыре contraction/token: outer `q*dO`, `dS@innovation`, `k.T@dS`,
outer `k*dV`. Затем прямой проход выполняет ещё три: `S@dV`, реконструкция
состояния outer product, `Snext@dO`. Core-backward = `B*H*T*14*d²`, а не
`2*forward`. Восстановление innovation через новый matvec не требуется.
Scalar `dbeta` reduction исключён по общей политике.

Для `M=B*T` четыре проекции `D -> D` имеют каждая forward `2*M*D²`, backward
`4*M*D²`. Beta-проекция `D -> H`: `2*M*D*H` и `4*M*D*H`.

Три short-conv дают forward `3*2*M*D*4`. Backward включает input-gradient,
weight-gradient **и повторный convolution preactivation для SiLU**. Поэтому
backward short-conv равен **трём**, а не двум forward-conv. Производная SiLU
сама исключена. CUDA/другие convolution backends не покрываются этим выводом.

## Независимая проверка

`tests/compare_attention/train/delta_reference.py` исполняет PyTorch chunk-WY
алгоритм: четыре diagonal row-solve, block inverse merge, W/U, chunk-state,
chunk output и отдельный ручной VJP в порядке FLA с рекомпутацией. `Ledger.mm`
получает стоимость из фактических форм операндов каждого вызова; он не вызывает
production estimator и не копирует его полиномиальные формулы.

Проверяются:

- Численная эквивалентность chunk-WY и recurrence, double precision.
  Recurrence — только oracle для значений, не основание для chunk FLOPs.
- Ручной chunk/recurrent VJP против независимого PyTorch autograd, все q/k/v/beta.
- Полное равенство каждого core-компонента shape-ledger и estimator.
- Native linear forward/autograd FLOPs; convolution через отдельные channel
  matmuls с двумя производными и одним preactivation recompute. Не используется
  grouped-conv backward formula Torch: она не учитывает groups для weight VJP.
- Algebraic B/T/d scaling, статические типы, unsupported shapes/versions/backends,
  dtype/compiler-flag/seed independence и сохранение CPU RNG.
- Реальные FLA CUDA outputs и q/k/v/beta gradients для T=17,64,128 BF16 против
  double reference; relative RMS error каждого тензора <4%.
- Реальный default-адаптер **B8,T512,D768,H12 BF16**: forward, backward, finite
  output/input-gradient/все parameter-gradients, backend short-conv Triton.

Локальная CUDA-проверка выполнена на RTX 5070, Torch 2.11.0+cu128. Четыре CUDA
теста прошли. Повторная совместная проверка DeltaNet и общего estimator:
`60 passed, 14 warnings in 9.14s`; Ruff check/format прошли, Pylint 10.00/10
на четырёх изменённых Python-файлах. Полный regression suite здесь не запускался.
Первый CUDA-запуск занял 158.88 s и выдал 222 сторонних DeprecationWarning
(Torch JIT и Triton/Python AST), то есть результат не warning-free. Эти тесты
не доказывают идентичность числа аппаратных инструкций и не являются Kaggle smoke.

## Числа для production shape

Для одного attention-адаптера B8,T512,D768,H12:

```text
forward  = 22,822,256,640
backward = 48,293,216,256
total    = 71,115,472,896 algorithmic FLOPs v2
```

Это не цена полного GPT train-step. Общий estimator должен умножить компоненты
attention на число слоёв и добавить собственные GPT dense-компоненты.

## Непокрытые случаи

Не поддерживаются tail chunk, пользовательские chunk sizes, gate/expand/default
изменения, masks/packing/cache, optimizer compute, checkpointing, frozen parameters,
нестандартные backends и другие FLA releases. Проверка compiler-флага estimator
не доказывает graph capture всего train-loop. Exact-one-step Kaggle smoke и
решение о запуске серии принадлежат общему orchestration gate; здесь cloud-действий
нет. При добавлении варианта необходимы новый source-audit и RED→GREEN тесты,
а не экстраполяция текущих коэффициентов.
