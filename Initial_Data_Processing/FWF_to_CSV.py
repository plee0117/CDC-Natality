def create_column_dictionary(text_file):
    '''
    Takes text file from User Guide to create library identifying locations of each feature.
    Returns pd.DataFrame
    The 'length' feature appears to be inconsistent
    '''

    import re
    import pandas as pd

    with open(text_file, 'r+') as f:
        ug = f.readlines()

    # Get field codes and corresponding columns
    field_list = [
        re.search(r'\b[0-9-]+\b\s\b[0-9]+\b\s\b[A-Z0-9_]{2,}\b',i).group() for i in ug if 
        re.search(r'\b[0-9-]+\b\s\b[0-9]+\b\s\b[A-Z0-9_]{2,}\b',i)
    ]

    # Place field numbers into DF
    field_code = pd.DataFrame(columns = ['start','end','length','field'])
    field_code['length'] = [re.split('\s',i)[1] for i in field_list]
    field_code['field'] = [re.split('\s',i)[2] for i in field_list]
    field_range = [re.split('\s',i)[0] for i in field_list]
    field_code['start'] = [re.split('-',i)[0] if re.search('-',i) else i for i in field_range]
    field_code['end'] = [re.split('-',i)[1] if re.search('-',i) else i for i in field_range]

    # Convert entry to numeric
    for i in ['start','end','length']:
        field_code[i] = pd.to_numeric(field_code[i])
    
    # Missing entries
    field_code_ug2018 = pd.DataFrame([[165,165,1,'F_FEDUC'], [280,281,2,'M_Ht_In'], 
                                      [292,294,3,'PWgt_R'], [299,301,3,'DWgt_R'], 
                                      [328,328,1,'f_RF_INFT'], [499,500,2,'OEGest_Comb'], 
                                      [501,502,2,'OEGest_R10'], [503,503,1,'OEGest_R3']], 
                                     columns = ['start','end','length','field'])
    
    field_code_ug2017 = pd.DataFrame([[280, 281, 2, 'M_Ht_In'],[292, 294, 3, 'PWgt_R'],[299, 301, 3, 'DWgt_R'],
                                      [328, 328, 1, 'f_RF_INFT'],[499, 500, 2, 'OEGest_Comb'],
                                      [501, 502, 2, 'OEGest_R10'],[503, 503, 1, 'OEGest_R3']],
                                     columns = ['start','end','length','field'])
    
    field_code_ug2016 = pd.DataFrame([[280, 281, 2, 'M_Ht_In'],[292, 294, 3, 'PWgt_R'],[299, 301, 3, 'DWgt_R'],
                                      [328, 328, 1, 'f_RF_INFT'],[499, 500, 2, 'OEGest_Comb'],
                                      [501, 502, 2, 'OEGest_R10'],[503, 503, 1, 'OEGest_R3']],
                                     columns = ['start','end','length','field'])
    
    field_code_ug2008 = pd.DataFrame([[609, 609, 1, 'F_LD_AUGMENT']],
                                     columns = ['start','end','length','field'])
    


    # Combine missing entries 
    if text_file.lower() == 'ug2018.txt':
        field_code = field_code.append(field_code_ug2018,ignore_index= True,sort = 'start')
        field_code = field_code.sort_values('start').reset_index().drop(columns='index')
    elif text_file.lower() == 'ug2017.txt':
        field_code = field_code.append(field_code_ug2017,ignore_index= True,sort = 'start')
        field_code = field_code.sort_values('start').reset_index().drop(columns='index')
    elif text_file.lower() == 'ug2016.txt':
        field_code = field_code.append(field_code_ug2016,ignore_index= True,sort = 'start')
        field_code = field_code.sort_values('start').reset_index().drop(columns='index')
    elif text_file.lower() == 'ug2008.txt':
        field_code = field_code.append(field_code_ug2008,ignore_index= True,sort = 'start')
        field_code = field_code.sort_values('start').reset_index().drop(columns='index')

    print(field_code.columns)
    

    # Find inconsistencies
    for i in range(0,len(field_code)-1):
        if field_code['end'][i] + 1 != field_code['start'][i+1]:
            print('check after: ', i,  field_code['field'][i], field_code['start'][i], '-', field_code['end'][i])
#         if field_code['end'][i] - field_code['start'][i] + 1 != field_code['length'][i]:
#             print('check range for: ', i,  field_code['field'][i], field_code['start'][i], '-', 
#                   field_code['end'][i], field_code['length'][i])

    

    return field_code


def no_more_filler(field_code):
    '''
    Looks for field that starts with 'FILLER' and drop the corresponding entry
    Returns shorter library pd.DataFrame
    '''
    import re

    field_code = field_code[[False if re.search('^FILLER',i) else True for i in field_code['field']]]
    field_code.reset_index(inplace = True)
    field_code.drop(columns = ['index'], inplace = True)
    return field_code

def convert_to_csv(text_file, csv_file, column_dictionary):
    '''
    Pass fwf text, destination csv, column dictionary
    Read one line of the text file into python at a time
    Break each row apart into list and write onto csv
    '''

    import csv
    import time
    import re
    
    start_at = time.time()
    j=0
    c_dict = column_dictionary
    with open(csv_file,'w') as c:
        wc = csv.writer(c,quoting=csv.QUOTE_MINIMAL)
        
        # Write header
        wc.writerow(list(c_dict['field']))

        with open(text_file, 'r+') as f:
            for line in f:

#                 items = ["" if re.search(r'\A\s+\Z',line[c_dict['start'][i]-1 : c_dict['end'][i]]) 
#                          else line[c_dict['start'][i]-1 : c_dict['end'][i]] 
#                          for i in range(0,len(c_dict))]
                items = []
                for i in range(0,len(c_dict)):
#                     print(i)
                    from_here = c_dict['start'][i]-1
                    to_here = c_dict['end'][i]
                    if re.search(r'\A\s+\Z', line[from_here:to_here]):
                        items += [""]
                    else:
                        items += [line[from_here:to_here]]
#                     print(from_here, to_here)

                wc.writerow(items)
#         keep track of progress
                j += 1
                if j>20:
                    break
#                 if j%10000 == 0:
#                     print(j, time.time() - start_at)
    print('done at',j, 'in', time.time() - start_at)


field_code = create_column_dictionary('UG2017.txt')
field_code = no_more_filler(field_code)
convert_to_csv('Nat2017PublicUS.c20180516.r20180808.txt','CSV2017Head.csv',field_code)
