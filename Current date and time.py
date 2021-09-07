# import datetime
#
# d = datetime.datetime.now()
#
# print("1: test-{date:%Y-%m-%d %H:%M:%S}.txt".format(date= datetime.datetime.now()))
#
#
# print("F:{date:%Y-%m-%d_%Hh%Mm%Ss}.h5".format(date= datetime.datetime.now()))
#
# print("2: {:%B %d, %Y}".format(d))
#
# print("3: Today is {datetime.datetime.now(): %Y-%d-%m} yay")
# #
# # Date = "M:{d:%Y-%m-%d_%Hh:%Mm:%Ss}".format(d=datetime.datetime())
# # print(Date)
#
# print("M:{datetime.datetime(): %Y-%m-%d_%Hh:%Mm:%Ss} yay")


import datetime
print(
    '1: test-{date:%Y-%m-%d %H:%M:%S}.txt'.format( date=datetime.datetime.now() )
    )

date = datetime.datetime.now()
print('2: {:%B %d, %Y}'.format(date))

print(f"3: Today is {datetime.datetime.now():%Y-%m-%d} yay")


print(f"M:{date:%Y-%m-%d_%Hh%Mm%Ss}")

