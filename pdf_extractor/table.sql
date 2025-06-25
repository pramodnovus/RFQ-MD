CREATE TABLE IF NOT EXISTS extracted_projects (
    id SERIAL PRIMARY KEY,
    name TEXT,
    target_group TEXT,
    loi TEXT,
    location TEXT,
    sample_size INTEGER,
    man_days INTEGER
);
