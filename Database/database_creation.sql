CREATE DATABASE IF NOT EXISTS final_proyect_ironhack;

USE final_proyect_ironhack;

CREATE TABLE IF NOT EXISTS Run (
    run_num INT PRIMARY KEY,
    date_run DATE
);

CREATE TABLE IF NOT EXISTS Particle_A (
    id_part INT AUTO_INCREMENT PRIMARY KEY,
    energy FLOAT NOT NULL,
    px FLOAT NOT NULL,
    py FLOAT NOT NULL,
    pz FLOAT NOT NULL,
    pt FLOAT NOT NULL,
    eta FLOAT NOT NULL,
    phi FLOAT NOT NULL,
    charge TINYINT NOT NULL
);

CREATE TABLE IF NOT EXISTS Particle_B (
    id_part INT AUTO_INCREMENT PRIMARY KEY,
    energy FLOAT NOT NULL,
    px FLOAT NOT NULL,
    py FLOAT NOT NULL,
    pz FLOAT NOT NULL,
    pt FLOAT NOT NULL,
    eta FLOAT NOT NULL,
    phi FLOAT NOT NULL,
    charge TINYINT NOT NULL
);

CREATE TABLE IF NOT EXISTS Event (
    id_event INT AUTO_INCREMENT PRIMARY KEY,
    event_num INT NOT NULL,
    run_num INT,
    id_partA INT NOT NULL,
    id_partB INT NOT NULL,
	invariant_mass FLOAT,
    FOREIGN KEY (run_num) REFERENCES Run(run_num) ON UPDATE CASCADE ON DELETE SET NULL,
    FOREIGN KEY (id_partA) REFERENCES Particle_A(id_part) ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (id_partB) REFERENCES Particle_B(id_part) ON UPDATE CASCADE ON DELETE CASCADE
);

