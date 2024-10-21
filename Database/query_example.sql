SELECT *
FROM Event e
INNER JOIN Particle_A a ON e.id_partA = a.id_part
INNER JOIN Particle_B b ON e.id_partB = b.id_part
WHERE e.event_num = 367112316;
