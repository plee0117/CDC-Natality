#Load the Schema, replace with your named schema
use CDC_baby;

#needed for loading
SHOW GLOBAL VARIABLES LIKE 'local_infile';

# generate the head from the created csv file in the command line
# HEAD CVS2018.csv > CSV2018head.csv
# load it through Table Import Wizard

ALTER TABLE CVS2018head MODIFY DMAR TEXT;

#after import wizarding the head file, view to see it's all there.
SELECT * FROM CVS2018head;

DROP TABLE IF EXISTS cdc_2018;

#Create a new table like CSV2018head
CREATE TABLE cdc_2018 LIKE CVS2018head;
    
#double check cdc_2018, just header
SELECT *
FROM cdc_2018;

#Import full dataset into the table
LOAD DATA LOCAL INFILE '/private/tmp/CSV2018.csv' IGNORE
INTO TABLE cdc_2018
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
#ESCAPED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

SELECT *
FROM cdc_2018;
