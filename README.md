create database olympixs;
use olympixs;


/* CREATE TABLE */
CREATE TABLE details(
id int primary key,
name VARCHAR(100),
age DOUBLE,
year DOUBLE,
Date_Given VARCHAR(100)
);


/* CREATE TABLE */
CREATE TABLE country(
country_id int ,
country VARCHAR(100),
id int,
PRIMARY KEY (country_id),
FOREIGN KEY (id) REFERENCES details(id)
);





/* CREATE TABLE */
CREATE TABLE medals(
sports_id int,
sports VARCHAR(100),
gold_medal DOUBLE,
silver_medal DOUBLE,
brone_medal DOUBLE,
total_medal DOUBLE,
id int,
PRIMARY KEY (sports_id),
FOREIGN KEY (id) REFERENCES details(id)
);
