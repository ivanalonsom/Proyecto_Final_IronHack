CREATE DATABASE IF NOT EXISTS fisica_particulas;


USE fisica_particulas;


CREATE TABLE IF NOT EXISTS Run (
    num_run INT PRIMARY KEY,
    fecha DATE
);


CREATE TABLE IF NOT EXISTS Particula_A (
    id_part INT PRIMARY KEY,
    energy FLOAT,
    px FLOAT,
    py FLOAT,
    pz FLOAT,
    pt FLOAT,
    eta FLOAT,
    phi FLOAT,
    charge INT
);


CREATE TABLE IF NOT EXISTS Particula_B (
    id_part INT PRIMARY KEY,
    energy FLOAT,
    px FLOAT,
    py FLOAT,
    pz FLOAT,
    pt FLOAT,
    eta FLOAT,
    phi FLOAT,
    charge INT
);


CREATE TABLE IF NOT EXISTS Evento (
    num_evento INT PRIMARY KEY,
    num_run INT,
    id_partA INT,
    id_partB INT,
    FOREIGN KEY (num_run) REFERENCES Run(num_run),
    FOREIGN KEY (id_partA) REFERENCES Particula_A(id_part),
    FOREIGN KEY (id_partB) REFERENCES Particula_B(id_part)
);
