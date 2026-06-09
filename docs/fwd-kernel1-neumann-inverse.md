# Kernel 1 Neumann Inverse Helper

This note explains the call in `csrc/smxx/fwd_kernel1.cuh`:

```cpp
neumann_inv_fused_1warp(L_fp16, INV_fp16, INV, compute_tid);
```

The helper itself is defined in `csrc/smxx/utils.cuh` as `neumann_inv_fused_1warp`.

## Reset The Idea

Start with the scalar analogy.

Suppose we want the inverse of:

```text
1 + x
```

There is an expansion:

```text
1 / (1 + x) = 1 - x + x^2 - x^3 + x^4 - ...
```

You can verify this by multiplying:

```text
(1 + x)(1 - x + x^2 - x^3)
```

Expand it:

```text
= 1 - x + x^2 - x^3
  + x - x^2 + x^3 - x^4
```

Everything cancels except:

```text
= 1 - x^4
```

So if:

```text
x^4 = 0
```

then:

```text
(1 + x)(1 - x + x^2 - x^3) = 1
```

That means:

```text
1 - x + x^2 - x^3
```

is the exact inverse.

## Matrix Version

Now replace scalar `x` with matrix `L`.

In this kernel:

```text
L = strictly lower-triangular 16x16 matrix
```

For this kind of matrix:

```text
L^16 = 0
```

So the inverse of:

```text
I + L
```

is exactly:

```text
I - L + L^2 - L^3 + L^4 - ... - L^15
```

This is not an infinite approximation here. It stops because `L^16 = 0`.

## Why Factorize

The code does not want to add all 16 terms one by one.

Instead, it uses this identity:

```text
I - L + L^2 - L^3 + ... - L^15
=
(I - L)(I + L^2)(I + L^4)(I + L^8)
```

Why this works:

```text
(I - L)(I + L^2)
= I + L^2 - L - L^3
= I - L + L^2 - L^3
```

Now we have terms up to `L^3`.

Then multiply by:

```text
(I + L^4)
```

So:

```text
(I - L + L^2 - L^3)(I + L^4)
```

expands to:

```text
I - L + L^2 - L^3
+ L^4 - L^5 + L^6 - L^7
```

Now we have terms up to `L^7`.

Then multiply by:

```text
(I + L^8)
```

and we get:

```text
I - L + L^2 - L^3 + ... + L^14 - L^15
```

That is the complete finite inverse series for a `16x16` strictly lower-triangular `L`.

## Map To The Code

Before the helper, the kernel creates:

```text
INV_fp16 = I - L
```

That is the first factor.

Inside `neumann_inv_fused_1warp`, the code computes:

```cpp
// L^2 = L x L
```

Then it does:

```cpp
// INV += INV x L^2
```

That means:

```text
INV = INV + INV * L^2
    = INV * (I + L^2)
```

Since `INV` started as:

```text
I - L
```

now it becomes:

```text
(I - L)(I + L^2)
= I - L + L^2 - L^3
```

Then the code computes:

```cpp
// L^4 = L^2 x L^2
```

Then:

```cpp
// INV += INV x L^4
```

So now:

```text
INV = INV * (I + L^4)
```

which expands from terms up to `L^3` into terms up to `L^7`.

Then the code computes:

```cpp
// L^8 = L^4 x L^4
```

Then:

```cpp
// INV += INV x L^8
```

So now:

```text
INV = INV * (I + L^8)
```

which expands from terms up to `L^7` into terms up to `L^15`.

Final result:

```text
INV = I - L + L^2 - L^3 + ... - L^15
```

That is:

```text
(I + L)^-1
```

## Short Version

The helper computes an inverse-like triangular correction by doing only:

```text
L^2
L^4
L^8
```

instead of explicitly computing every power from `L^1` to `L^15`.

The key one-liner:

```text
Start with I - L, then multiply in (I + L^2), (I + L^4), and (I + L^8).
That expands to the full finite inverse series.
```
