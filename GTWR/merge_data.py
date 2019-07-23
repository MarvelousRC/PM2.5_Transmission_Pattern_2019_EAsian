if __name__ == '__main__':
    file_name_first = r'.\table\data-'
    date_str_list = ['14', '15', '16']
    utc_str_list = ['1', '4', '7']
    text_all = 'FID,city,lat,lon,pm2_5,AOD,AirTemp,DPTemp,RH,SeaLevelPr,WindDir,WindSpeed,NDVI,DEM,POINT_X,POINT_Y,time\n'
    for date in date_str_list:
        for utc in utc_str_list:
            file_name = file_name_first + date + '-' + utc + '.csv'
            file = open(file_name, 'r')
            lines = file.read().splitlines()
            for i in range(len(lines)):
                if i != 0:
                    text_all += lines[i] + ',' + str(float(date) + float(utc) / 24) + '\n'
            file.close()
    file = open(r'.\table\final_data.csv', 'w')
    file.write(text_all)
    file.close()
    print('Merge data finished!')
