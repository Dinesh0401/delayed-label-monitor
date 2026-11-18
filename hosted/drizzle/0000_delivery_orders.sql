CREATE TABLE orders (id TEXT PRIMARY KEY, state TEXT NOT NULL,
 purchased TEXT NOT NULL, due TEXT NOT NULL, probability REAL NOT NULL,
 predicted INTEGER NOT NULL, model_version TEXT NOT NULL, created TEXT NOT NULL,
 delivered TEXT, reported TEXT);
