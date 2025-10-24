DROP TABLE IF EXISTS metrics;

-- create enum for dataset type
DROP TYPE IF EXISTS dataset_split;
CREATE TYPE dataset_split AS ENUM ('train', 'validate', 'test');


-- metrics table
CREATE TABLE metrics (
    ID INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id INTEGER NOT NULL,
    model TEXT NOT NULL,
    data dataset_split NOT NULL,
    threshold_notes TEXT,
    pipeline_notes TEXT,
    hyperparam_notes TEXT,
    notes TEXT,
    roc_auc DOUBLE PRECISION,
    accuracy DOUBLE PRECISION DEFAULT 0.0,
    precision DOUBLE PRECISION DEFAULT 0.0,
    recall DOUBLE PRECISION DEFAULT 0.0,
    f1 DOUBLE PRECISION DEFAULT 0.0,
    TN INTEGER DEFAULT 0,
    TP INTEGER DEFAULT 0,
    FP INTEGER DEFAULT 0,
    FN INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);



-- 
DROP TABLE IF EXISTS hyperparameters;

CREATE TABLE hyperparameters (
    ID INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id INTEGER NOT NULL,
    model TEXT NOT NULL,
    hyperparam_text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (run_id, model)
);