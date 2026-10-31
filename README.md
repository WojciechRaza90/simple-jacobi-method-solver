# simple-jacobi-method-solver

Iteratively solves a diagonally dominant system of linear equations `A x = b` using the Jacobi method.

## Usage

```js
import { jacobi } from './src/index.js';

const A = [
  [4, 1],
  [1, 3],
];
const b = [1, 2];

const x = jacobi(A, b);
console.log(x); // approximately [0.0909, 0.6364]
```

With options:

```js
import { jacobi } from './src/index.js';

const A = [
  [4, 1],
  [1, 3],
];
const b = [1, 2];

const x = jacobi(A, b, {
  maxIterations: 500,
  tolerance: 1e-8,
  initialGuess: [0, 0],
});
```

## Why this exists

The Jacobi method is the simplest iterative solver for linear systems. It is rarely the fastest, but it is trivial to implement correctly and has no dependencies. This library exists for cases where pulling in a numerical linear algebra package is overkill and the matrix is small and diagonally dominant.

The trade-off: we do not attempt to reorder rows or columns to improve diagonal dominance. If the matrix is not diagonally dominant, the iteration may diverge, and we throw `NonConvergenceError` rather than trying to salvage it. This keeps the implementation small and predictable.

## Edge cases

- The diagonal of `A` must be non-zero. If any `A[i][i]` is zero, `SingularMatrixError` is thrown.
- `A` must be square and `b` must have matching length.
- If the method does not reach `tolerance` within `maxIterations`, `NonConvergenceError` is thrown with the final residual.
- Convergence is checked using the infinity norm of the residual `||A x - b||_inf`.

## API

### `jacobi(A, b, options?)`

Returns the solution vector `x`.

- `A`: `number[][]` — square matrix.
- `b`: `number[]` — right-hand side vector.
- `options.maxIterations` (default `1000`)
- `options.tolerance` (default `1e-10`)
- `options.initialGuess` (default zeros)

Throws `SingularMatrixError` for invalid input, `NonConvergenceError` if not converged.

### Errors

- `SingularMatrixError`
- `NonConvergenceError`
