/**
 * Local test and preview adapter providing the Cloudflare D1 subset for node:sqlite.
 *
 * @module db-adapter
 */

/**
 * Wraps a synchronous SQLite database instance into an asynchronous D1-compatible interface.
 *
 * @param {import('node:sqlite').DatabaseSync} sqlite - Synchronous SQLite database.
 * @returns {{prepare: (sql: string) => object}} D1-compatible database adapter.
 */
export function adapter(sqlite) {
  return {
    prepare(sql) {
      let args = [];
      const stmt = {
        bind(...values) {
          args = values;
          return stmt;
        },
        async run() {
          return sqlite.prepare(sql).run(...args);
        },
        async first() {
          return sqlite.prepare(sql).get(...args) || null;
        },
        async all() {
          return { results: sqlite.prepare(sql).all(...args) };
        },
      };
      return stmt;
    },
  };
}
