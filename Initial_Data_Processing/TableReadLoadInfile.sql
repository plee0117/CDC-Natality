#Load the Schema, replace with your named schema
#and make sure your my.cnf file has
-- [client]
-- port		= 3306
-- [mysqld]
-- port		= 3306
-- secure_file_priv=''
-- local-infile = 1
-- innodb_strict_mode = 0


-- SET GLOBAL innodb_file_per_table=ON;
SET GLOBAL innodb_log_buffer_size = 80*1024*1024*1024;
SET GLOBAL innodb_strict_mode = 0;
SET sql_mode = '';

use CDC;

#double check if local infile is ON
SHOW GLOBAL VARIABLES LIKE 'local_infile';

# generate the head from the created csv file in the command line
# HEAD CSV018.csv > CSV2018head.csv or use the text only/single line headercsv
#head -n NUMBEROFLINES file.csv > mynewfile.csv
# load it through Table Import Wizard

ALTER TABLE csvhead2018 MODIFY DMAR TEXT;

#check types in the columns
SELECT DATA_TYPE 
FROM INFORMATION_SCHEMA.COLUMNS
WHERE 
     TABLE_NAME = 'csvhead2018';


#after import wizarding the head file, view to see it's all there.
SELECT * FROM csvhead2018;

#Create a new table like CSV2018head, don't mind error
CREATE TABLE cdc2018 LIKE csvhead2018;
    
-- DROP TABLE cdc2018; 
#double check cdc_2018, just header
SELECT *
FROM cdc2018;

#check types in the columns
SELECT DATA_TYPE 
FROM INFORMATION_SCHEMA.COLUMNS
WHERE 
     TABLE_NAME = 'cdc2018';

#Import full dataset into the table - replace with your dataset name
#you will get an error, but it will WORK for some reason
LOAD DATA LOCAL INFILE '/private/tmp/CSV2018.csv' IGNORE
INTO TABLE cdc2018
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES ;

#double check inside
SELECT *
FROM cdc2018;

#double check rows
SELECT count(*)
FROM cdc2018;

#double check columns
SELECT COUNT(*)
  FROM INFORMATION_SCHEMA.COLUMNS
 WHERE table_name = 'cdc2018'


