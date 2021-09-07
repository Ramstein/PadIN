# import pickle, os
# i=1
# x = ['ok', 'but', 'this ','is','not funny']
# for str in x:
#     if i>=2:
#         os.remove('{}_pickle'.format(i-1))
#     pickle_out = open('{}_pickle'.format(i), "wb")
#     pickle.dump(str, pickle_out)
#     pickle_out.close()
#     i+=1
#
# pickle_upload = open('{}_pickle'.format(i-1), 'rb')
# p_upload = pickle.load(pickle_upload)
# print(p_upload)



x = .026262651616112165
print(round(x, 4))