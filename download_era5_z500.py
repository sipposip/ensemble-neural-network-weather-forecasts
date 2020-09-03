import cdsapi

c = cdsapi.Client()
var='geopotential'
plev='500'
startyear=1979
endyear=2018

path='/proj/bolinc/users/x_sebsc/nn_reanalysis/era5/'
years= [str(yr) for yr in range(startyear, endyear+1)]

for year in years:
    print(f'retrieving year {year}')

    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type':'reanalysis',
            'variable':var,
            'pressure_level':plev,
            'year':year,
            'format':'netcdf',
            'month':[
                '01','02','03',
                '04','05','06',
                '07','08','09',
                '10','11','12'
            ],
            'day':[
                '01','02','03',
                '04','05','06',
                '07','08','09',
                '10','11','12',
                '13','14','15',
                '16','17','18',
                '19','20','21',
                '22','23','24',
                '25','26','27',
                '28','29','30',
                '31'
            ],
            'time':[
                '00:00','01:00','02:00',
                '03:00','04:00','05:00',
                '06:00','07:00','08:00',
                '09:00','10:00','11:00',
                '12:00','13:00','14:00',
                '15:00','16:00','17:00',
                '18:00','19:00','20:00',
                '21:00','22:00','23:00'
            ]
        },
        f'{path}/era5_{var}{plev}_{year}.nc')
