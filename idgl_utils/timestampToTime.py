"""
    日期时间转时间戳函数 和 时间戳转日期时间函数
"""

import datetime

def day28(d):
    return d-28

def day29(d):
    return d-29

def day30(d):
    return d-30

def day31(d):
    return d-31

def whichMonth(d,runTorF):
    if runTorF:
        if day31(d) < 0:
            day = d
            return 1,day
        elif day29(day31(d)) < 0:
            day = day31(d)
            return 2,day
        elif day31(day29(day31(d))) < 0:
            day = day29(day31(d))
            return 3,day
        elif day30(day31(day29(day31(d)))) < 0:
            day = day31(day29(day31(d)))
            return 4,day
        elif day31(day30(day31(day29(day31(d))))) < 0:
            day = day30(day31(day29(day31(d))))
            return 5,day
        elif day30(day31(day30(day31(day29(day31(d)))))) < 0:
            day = day31(day30(day31(day29(day31(d)))))
            return 6,day
        elif day31(day30(day31(day30(day31(day29(day31(d))))))) < 0:
            day = day30(day31(day30(day31(day29(day31(d))))))
            return 7,day
        elif day31(day31(day30(day31(day30(day31(day29(day31(d)))))))) < 0:
            day = day31(day30(day31(day30(day31(day29(day31(d)))))))
            return 8,day
        elif day30(day31(day31(day30(day31(day30(day31(day29(day31(d))))))))) < 0:
            day = day31(day31(day30(day31(day30(day31(day29(day31(d))))))))
            return 9,day
        elif day31(day30(day31(day31(day30(day31(day30(day31(day29(day31(d)))))))))) < 0:
            day = day30(day31(day31(day30(day31(day30(day31(day29(day31(d)))))))))
            return 10,day
        elif day30(day31(day30(day31(day31(day30(day31(day30(day31(day29(day31(d))))))))))) < 0:
            day = day31(day30(day31(day31(day30(day31(day30(day31(day29(day31(d))))))))))
            return 11,day
        elif day31(day30(day31(day30(day31(day31(day30(day31(day30(day31(day29(day31(d)))))))))))) < 0:
            day = day30(day31(day30(day31(day31(day30(day31(day30(day31(day29(day31(d)))))))))))
            return 12,day
    else:
        if day31(d) < 0:
            day = d
            return 1, day
        elif day28(day31(d)) < 0:
            day = day31(d)
            return 2, day
        elif day31(day28(day31(d))) < 0:
            day = day28(day31(d))
            return 3, day
        elif day30(day31(day28(day31(d)))) < 0:
            day = day31(day28(day31(d)))
            return 4, day
        elif day31(day30(day31(day28(day31(d))))) < 0:
            day = day30(day31(day28(day31(d))))
            return 5, day
        elif day30(day31(day30(day31(day28(day31(d)))))) < 0:
            day = day31(day30(day31(day28(day31(d)))))
            return 6, day
        elif day31(day30(day31(day30(day31(day28(day31(d))))))) < 0:
            day = day30(day31(day30(day31(day28(day31(d))))))
            return 7, day
        elif day31(day31(day30(day31(day30(day31(day28(day31(d)))))))) < 0:
            day = day31(day30(day31(day30(day31(day28(day31(d)))))))
            return 8, day
        elif day30(day31(day31(day30(day31(day30(day31(day28(day31(d))))))))) < 0:
            day = day31(day31(day30(day31(day30(day31(day28(day31(d))))))))
            return 9, day
        elif day31(day30(day31(day31(day30(day31(day30(day31(day28(day31(d)))))))))) < 0:
            day = day30(day31(day31(day30(day31(day30(day31(day28(day31(d)))))))))
            return 10, day
        elif day30(day31(day30(day31(day31(day30(day31(day30(day31(day28(day31(d))))))))))) < 0:
            day = day31(day30(day31(day31(day30(day31(day30(day31(day28(day31(d))))))))))
            return 11, day
        elif day31(day30(day31(day30(day31(day31(day30(day31(day30(day31(day28(day31(d)))))))))))) < 0:
            day = day30(day31(day30(day31(day31(day30(day31(day30(day31(day28(day31(d)))))))))))
            return 12, day

def timestampToDate(timestamp):
    timestamp = timestamp + 24 # 由于1900是非闰年
    hour = timestamp % 24
    timestamp = timestamp - hour
    fourYearEpoch = int(timestamp / 35064)
    dayMod = int(timestamp % 35064)
    dayMod = dayMod / 24
    if dayMod - 366 < 0:
        year = 1900 + fourYearEpoch * 4
        month,day = whichMonth(dayMod,True)
        day = int(day) + 1
        print(year,'-',month,'-',day,'  ',hour,':00')
    elif dayMod - (366 + 365) < 0:
        year = 1900 + fourYearEpoch * 4 + 1
        dayMod = dayMod - 366
        month, day = whichMonth(dayMod, False)
        day = int(day) + 1
        print(year, '-', month, '-', day, '  ', hour, ':00')
    elif dayMod - (366 + 365 + 365) < 0:
        year = 1900 + fourYearEpoch * 4 + 2
        dayMod = dayMod - 366 - 365
        month, day = whichMonth(dayMod, False)
        day = int(day) + 1
        print(year, '-', month, '-', day, '  ', hour, ':00')
    elif dayMod - (366 + 365 + 365 + 365) < 0:
        year = 1900 + fourYearEpoch * 4 + 3
        dayMod = dayMod - 366 - 365 - 365
        month, day = whichMonth(dayMod, False)
        day = int(day) + 1
        print(year, '-', month, '-', day, '  ', hour, ':00')


def timeToTimestamp(year,month,day,hour):
    d1 = datetime.date(1900, 1, 1)
    d2 = datetime.date(year, month, day)
    timestamp = (d2 - d1).days * 24 + hour
    return timestamp