import sqlite3
import logging
import json
import numpy as np
import pandas as pd
from itertools import repeat
from pathlib import Path

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger('base')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger.setLevel(logging.DEBUG)

RECALL_TARGETS = [0.8, 0.85, 0.9, 0.95, 0.99, 1.0]

pd.options.display.max_columns = 650
pd.options.display.max_rows = 20

path_rankings = Path('../data/converted')
path_target = Path('../data/rankings.sqlite')
path_target.unlink(missing_ok=True)

con = sqlite3.connect(path_target)
#cur = con.cursor()

logger.info('Populating simulation/ranking tables...')
con.executescript(
    '''
    -- Table to hold dataset descriptors
    CREATE TABLE dataset
    (
        dataset_id  INT         NOT NULL,
        name        VARCHAR(60) NOT NULL,
        description TEXT,
        n_incl      INT         NOT NULL,
        n_total     INT         NOT NULL,

        PRIMARY KEY (dataset_id)
    );

    -- In-between relation for each ranking simulation of a dataset
    CREATE TABLE simulation
    (
        simulation_id INT NOT NULL,
        dataset_id    INT NOT NULL,
        repeat        INT NOT NULL,

        PRIMARY KEY (simulation_id),
        UNIQUE (dataset_id, repeat),
        FOREIGN KEY (dataset_id) REFERENCES dataset (dataset_id)
    );

    -- Machine-learning model info per ranking batch
    CREATE TABLE simulation_batches
    (
        simulation_id INT NOT NULL,
        batch         INT NOT NULL,
        recall        REAL,
        precision     REAL,
        f1            REAL,
        batch_size    INT NOT NULL,
        params        JSONB,

        PRIMARY KEY (simulation_id, batch),
        FOREIGN KEY (simulation_id) REFERENCES simulation (simulation_id)
    );

    -- The actual ranking (gold-standard label and predicted score per record) with reference to the simulation batch
    -- also includes convenience columns (seen, seen incl)
    CREATE TABLE ranking
    (
        simulation_id INT     NOT NULL,
        record_id     INT     NOT NULL,
        label         BOOLEAN NOT NULL,
        score         REAL,
        batch         INT     NOT NULL,
        n_seen        INT     NOT NULL,
        n_seen_incl   INT     NOT NULL,

        PRIMARY KEY (simulation_id, record_id),
        FOREIGN KEY (simulation_id) REFERENCES simulation (simulation_id),
        FOREIGN KEY (simulation_id, batch) REFERENCES simulation_batches (simulation_id, batch)
    );

    -- Relation containing stopping method configurations
    CREATE TABLE stopping_method
    (
        method_id INT         NOT NULL,
        name      VARCHAR(60) NOT NULL,
        params    JSONB,

        PRIMARY KEY (method_id)
    );

    -- Relation containing (human) stopping decisions
    CREATE TABLE users
    (
        user_id    INT          NOT NULL,
        name       VARCHAR(60)  NOT NULL,
        email      VARCHAR(120) NOT NULL,
        created_at TIMESTAMP    NOT NULL default CURRENT_TIMESTAMP,
        updated_at TIMESTAMP,

        PRIMARY KEY (user_id)
    );
    CREATE TRIGGER users_update
        AFTER UPDATE
        ON users
        FOR EACH ROW BEGIN UPDATE users SET updated_at = CURRENT_TIMESTAMP WHERE user_id = old.user_id;
    END;

    CREATE TABLE session
    (
        session_id         INT       NOT NULL,
        user_id            INT       NOT NULL,

        settings           JSONB,
        data               JSONB,
        is_session_started BOOLEAN   NOT NULL DEFAULT FALSE,
        is_session_ended   BOOLEAN   NOT NULL DEFAULT FALSE,
        created_at         TIMESTAMP NOT NULL default CURRENT_TIMESTAMP,
        updated_at         TIMESTAMP,

        FOREIGN KEY (user_id) REFERENCES users (user_id)
    );
    CREATE TRIGGER session_update
        AFTER UPDATE
        ON session
        FOR EACH ROW BEGIN UPDATE session SET updated_at = CURRENT_TIMESTAMP WHERE session_id = old.session_id;
    END;

    CREATE TABLE stopping_decision
    (
        decision_id   INT NOT NULL,
        simulation_id INT NOT NULL,
        method_id     INT,
        user_id       INT,
        session_id    INT,
        n_seen        INT NOT NULL, -- index of where in the ranking this criterion stopped
        created_at    TIMESTAMP,
        params        JSONB,        -- additional information around the stopping decision

        PRIMARY KEY (decision_id),
        FOREIGN KEY (simulation_id) REFERENCES simulation (simulation_id),
        FOREIGN KEY (method_id) REFERENCES stopping_method (method_id),
        FOREIGN KEY (user_id) REFERENCES users (user_id),
        FOREIGN KEY (session_id) REFERENCES session (session_id),
        FOREIGN KEY (simulation_id, n_seen) REFERENCES ranking (simulation_id, n_seen),
        -- has to link to either a method or a user
        CHECK ((method_id IS NOT NULL AND user_id IS NULL) OR (method_id IS NULL AND user_id IS NOT NULL)),
        -- if this is a user stopping decision, it has to have a date
        CHECK (user_id IS NULL OR (user_id IS NOT NULL AND created_at IS NOT NULL))
    );
    ''',
)

LOG_PARAM_KEYS = {
    'APRIORI': {'est_recall'},
    'BATCHPRECISION': {'current_precision'},
    'BUSCAR': set([]),
    'CMH': set([]),
    'CURVE_FITTING': {'expected_includes', 'curve_estimate'},
    'HEURISTIC_FIX': set([]),
    'HEURISTIC_FRAC': {'num_to_stop'},
    'HEURISTIC_RANDOM': {'est_incl'},
    'HEURISTIC_SCORES': {'est_incl'},
    'IPP': {'est_incl'},
    'KNEE': {'slope_ratio'},
    'METHOD2399': {'num_reviewed', 'num_relevant_reviewed', 'threshold'},
    'QUANT_CI': {'est_recall', 'est_var'},
    'SALτ': {'est_recall', 'margin_recall'},
    'S-CAL': {'est_incl'},
    'TM_QBCB': {'n_sample', 'required_overlap', 'n_overlap'},
}

methods = {}
simulation_id = 0
decision_id = 0

for dataset_id, file in enumerate(path_rankings.glob('*.json')):
    logger.info(f'Processing dataset {dataset_id} from {file.name}')
    with open(file, 'r') as f:
        dataset = json.load(f)
        con.execute(
            'INSERT INTO dataset(dataset_id, name, n_incl, n_total) VALUES(?, ?, ?, ?)',
            (dataset_id, dataset['name'], dataset['n_incl'], dataset['n_total']),
        )

        for simulation in dataset['simulations']:
            simulation_id += 1
            con.execute(
                'INSERT INTO simulation(simulation_id, dataset_id, repeat) VALUES(?, ?, ?)',
                (simulation_id, dataset_id, simulation['ranking_info']['repeat']),
            )

            con.executemany(
                'INSERT INTO simulation_batches(simulation_id, batch, recall, precision, f1, batch_size, params) VALUES(?, ?, ?, ?, ?, ?, ?)',
                [
                    (simulation_id, batch['batch_i'], batch.get('recall'), batch.get('precision'), batch.get('f1'), batch['batch_size'], json.dumps(batch['params']))
                    for batch in simulation['ranking_info']['batches']
                ],
            )

            con.executemany(
                'INSERT INTO ranking(simulation_id, record_id, label, score, batch, n_seen, n_seen_incl) VALUES(?, ?, ?, ?, ?, ?, ?)',
                list(
                    zip(
                        repeat(simulation_id),
                        simulation['ranking']['id'],
                        simulation['ranking']['labels'],
                        [s if s >=0 else None for s in simulation['ranking']['score']],
                        simulation['ranking']['batch'],
                        range(1, len(simulation['ranking']['id'])),
                        np.cumsum(simulation['ranking']['labels']).astype(int).tolist(),
                    ),
                ),
            )

            for decision in simulation['stop_decisions']:
                decision_id += 1
                method_params = {k: v for k, v in decision['params'].items() if k not in LOG_PARAM_KEYS[decision['method']]}
                log_params = {k: v for k, v in decision['params'].items() if k in LOG_PARAM_KEYS[decision['method']]}
                method_key = f'{decision['method']}-{json.dumps(method_params)}'
                if method_key in methods:
                    method_id = methods[method_key]
                else:
                    method_id = len(methods)
                    methods[method_key] = method_id
                    con.execute(
                        'INSERT INTO stopping_method(method_id, name, params) VALUES(?, ?, ?)',
                        (method_id, decision['method'], json.dumps(method_params)),
                    )

                con.execute(
                    'INSERT INTO stopping_decision(decision_id, simulation_id, method_id, n_seen, params) VALUES(?, ?, ?, ?, ?)',
                    (decision_id, simulation_id, method_id, decision['n_seen'], json.dumps(log_params)),
                )

    con.commit()
con.close()


# SELECT *, json(params)->>'key'
# FROM simulation_batches;
#
# SELECT name, json_group_array(method_id), json_group_array(json(params))
# FROM stopping_method
# GROUP BY name;
