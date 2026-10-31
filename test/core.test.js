import { test } from 'node:test';
import assert from 'node:assert/strict';
import { jacobi, NonConvergenceError, SingularMatrixError } from '../src/index.js';

const EPS = 1e-9;

function assertVectorClose(actual, expected, eps = EPS) {
  assert.equal(actual.length, expected.length, 'vector length mismatch');
  for (let i = 0; i < actual.length; i++) {
    assert.ok(
      Math.abs(actual[i] - expected[i]) < eps,
      `index ${i}: got ${actual[i]}, expected ${expected[i]}`
    );
  }
}

test('solves a simple 2x2 diagonally dominant system', () => {
  // 4x + y = 1
  // x + 3y = 2
  // solution: x = 1/11, y = 7/11
  const A = [
    [4, 1],
    [1, 3],
  ];
  const b = [1, 2];
  const x = jacobi(A, b);
  assertVectorClose(x, [1 / 11, 7 / 11]);
});

test('solves a 3x3 diagonally dominant system', () => {
  // 4x - y + z = 7
  // 4x - 8y + z = -21
  // -2x + y + 5z = 15
  // solution: x = 2, y = 4, z = 3
  const A = [
    [4, -1, 1],
    [4, -8, 1],
    [-2, 1, 5],
  ];
  const b = [7, -21, 15];
  const x = jacobi(A, b);
  assertVectorClose(x, [2, 4, 3]);
});

test('respects an initial guess', () => {
  const A = [
    [2, 1],
    [1, 2],
  ];
  const b = [3, 3];
  const x = jacobi(A, b, { initialGuess: [1, 1] });
  assertVectorClose(x, [1, 1]);
});

test('throws SingularMatrixError for non-square matrix', () => {
  const A = [
    [1, 2],
    [3, 4],
    [5, 6],
  ];
  const b = [1, 2, 3];
  assert.throws(() => jacobi(A, b), SingularMatrixError);
});

test('throws SingularMatrixError for zero on diagonal', () => {
  const A = [
    [0, 1],
    [1, 2],
  ];
  const b = [1, 2];
  assert.throws(() => jacobi(A, b), SingularMatrixError);
});

test('throws SingularMatrixError for mismatched b length', () => {
  const A = [
    [2, 1],
    [1, 2],
  ];
  const b = [1, 2, 3];
  assert.throws(() => jacobi(A, b), SingularMatrixError);
});

test('throws NonConvergenceError when maxIterations is too small', () => {
  const A = [
    [2, 1],
    [1, 2],
  ];
  const b = [3, 3];
  assert.throws(
    () => jacobi(A, b, { maxIterations: 1, tolerance: 1e-15 }),
    NonConvergenceError
  );
});

test('converges with a looser tolerance', () => {
  const A = [
    [2, 1],
    [1, 2],
  ];
  const b = [3, 3];
  const x = jacobi(A, b, { tolerance: 1e-6 });
  assertVectorClose(x, [1, 1], 1e-5);
});

test('handles a 1x1 system', () => {
  const A = [[5]];
  const b = [10];
  const x = jacobi(A, b);
  assertVectorClose(x, [2]);
});

test('returns the same result regardless of initial guess for a convergent system', () => {
  const A = [
    [4, 1],
    [1, 3],
  ];
  const b = [1, 2];
  const x1 = jacobi(A, b, { initialGuess: [0, 0] });
  const x2 = jacobi(A, b, { initialGuess: [100, -100] });
  assertVectorClose(x1, x2, 1e-6);
});

test('throws when initialGuess has wrong length', () => {
  const A = [
    [2, 1],
    [1, 2],
  ];
  const b = [3, 3];
  assert.throws(() => jacobi(A, b, { initialGuess: [1] }), SingularMatrixError);
});

test('throws when A is empty', () => {
  assert.throws(() => jacobi([], []), SingularMatrixError);
});
