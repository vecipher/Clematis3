export const stableSort = <T>(arr: T[], key: (t: T) => string | number): T[] =>
  arr.map((v, i) => ({v, i}))
     .sort((a, b) => {
       const ka = key(a.v), kb = key(b.v);
       return ka < kb ? -1 : ka > kb ? 1 : a.i - b.i;
     })
     .map(x => x.v);
