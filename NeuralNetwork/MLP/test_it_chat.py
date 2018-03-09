import itchat
itchat.auto_login()
for friend in itchat.get_friends(update=True)[0:]:
    #可以用此句print查看好友的微信名、备注名、性别、省份、个性签名（1：男 2：女 0：性别不详）
    print(friend['NickName'],friend['RemarkName'],friend['Sex'],friend['Province'],friend['Signature'])
    # img = itchat.get_head_img(userName=friend["UserName"])
    # path = "/Users/xxx/HeadImages/"+friend['NickName']+"("+friend['RemarkName']+").jpg"
    # try:
    #     with open(path,'wb') as f:
    #         f.write(img)
    # except Exception as e:
    #     print(repr(e))
itchat.run()