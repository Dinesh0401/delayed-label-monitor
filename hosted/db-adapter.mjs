// Local test/preview adapter for the D1 subset used by this app.
export function adapter(sqlite){return {prepare(sql){let args=[];const stmt={bind(...values){args=values;return stmt},async run(){return sqlite.prepare(sql).run(...args)},async first(){return sqlite.prepare(sql).get(...args)||null},async all(){return {results:sqlite.prepare(sql).all(...args)}}};return stmt}}}
