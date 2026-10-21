// Wall-clock time for reporting how long the solver worked. A film render
// replaces performance.now with a virtual clock; the real one stays in __realNow.
export const now = () => (globalThis.__realNow ? globalThis.__realNow() : performance.now());
