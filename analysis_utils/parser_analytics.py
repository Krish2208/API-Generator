import re
import json

ip_reg = re.compile('''\d+\.\d+\.\d+\.\d+''')
timestamp_reg1 = re.compile('''((((19|20)([2468][048]|[13579][26]|0[48])|2000)-02-29|((19|20)[0-9]{2}-(0[4678]|1[02])-(0[1-9]|[12][0-9]|30)|(19|20)[0-9]{2}-(0[1359]|11)-(0[1-9]|[12][0-9]|3[01])|(19|20)[0-9]{2}-02-(0[1-9]|1[0-9]|2[0-8])))\s([01][0-9]|2[0-3]):([012345][0-9]):([012345][0-9]))''')
date_reg = re.compile('''^(?:(?:1[6-9]|[2-9]\d)?\d{2})(?:(?:(\/|-|\.)(?:0?[13578]|1[02])\1(?:31))|(?:(\/|-|\.)(?:0?[13-9]|1[0-2])\2(?:29|30)))$|^(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00)))(\/|-|\.)0?2\3(?:29)$|^(?:(?:1[6-9]|[2-9]\d)?\d{2})(\/|-|\.)(?:(?:0?[1-9])|(?:1[0-2]))\4(?:0?[1-9]|1\d|2[0-8])$''')
iso_format_reg = re.compile("(?:\\d{4})-(?:\\d{2})-(?:\\d{2})T(?:\\d{2}):(?:\\d{2}):(?:\\d{2}(?:\\.\\d*)?)(?:(?:-(?:\\d{2}):(?:\\d{2})|Z)?)")
email_reg = re.compile("(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x21\\x23-\\x5b\\x5d-\\x7f]|\\\\[\\x01-\\x09\\x0b\\x0c\\x0e-\\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x21-\\x5a\\x53-\\x7f]|\\\\[\\x01-\\x09\\x0b\\x0c\\x0e-\\x7f])+)\\])")
url_reg = re.compile("[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)")
mac_address_reg = re.compile("(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})")

def mysql_log(filepath):
    with open(filepath, 'r') as f:
        txt = f.read()

    complete_data = []
    for log in txt.split('\n')[1:]:
        data = dict()
        log = log.strip()
        query = log.split('\t')[-1]
        data['text'] = log
        entities = []
        ips = ip_reg.findall(log)
        if len(ips)>0:
            entities.append((log.index(ips[0]), log.index(ips[0])+len(ips[0]), 'IP_ADDRESS'))
        # tss = timestamp_reg1.findall(log)
        # if len(tss)>0:
        #     entities.append((log.index(tss[0]), log.index(tss[0])+len(tss[0]), 'TIMESTAMP'))
        isotss = iso_format_reg.findall(log)
        if len(isotss)>0:
            entities.append((log.index(isotss[0]), log.index(isotss[0])+len(isotss[0]), 'TIMESTAMP'))
        dates = date_reg.findall(log)
        if len(dates)>0:
            entities.append((log.index(dates[0]), log.index(dates[0])+len(dates[0]), 'DATE'))
        emails = email_reg.findall(log)
        if len(emails)>0:
            entities.append((log.index(emails[0]), log.index(emails[0])+len(emails[0]), 'EMAIL'))
        urls = url_reg.findall(log)
        if len(urls)>0:
            entities.append((log.index(urls[0]), log.index(urls[0])+len(urls[0]), 'URL'))
        macs = mac_address_reg.findall(log)
        if len(macs)>0:
            entities.append((log.index(macs[0]), log.index(macs[0])+len(macs[0]), 'MAC_ADDRESS'))
        entities.append((log.index(query), log.index(query)+len(query), 'QUERY'))
        data['entities'] = entities
        complete_data.append(data)
    return complete_data

def apache_log(filepath):
    with open(filepath, 'r') as f:
        txt = f.read()

    complete_data = []
    for log in txt.split('\n'):
        try:
            data = dict()
            log = log.strip()
            data['text'] = log
            entities = []
            res = re.findall(r'\[.*?\]',log)
            entities.append((log.index(res[0][1:-1]), log.index(res[0][1:-1])+len(res[0][1:-1]), "TIMESTAMP"))
            if "error" in log.lower():
                error_msg = log.split(']')[-1][1:]
                entities.append((log.index(error_msg), log.index(error_msg)+len(error_msg), "ERROR"))
            else:
                msg = log.split(']')[-1][1:]
                entities.append((log.index(msg), log.index(msg)+len(msg), "MESSAGE"))
            ips = ip_reg.findall(log)
            if len(ips)>0:
                # print(log, ips)
                entities.append((log.index(ips[0]), log.index(ips[0])+len(ips[0]), 'IP_ADDRESS'))
            data['entities'] = entities
            complete_data.append(data)
        except:
            continue
    return complete_data

def web_log(filepath):
    with open(filepath, 'r') as f:
        txt = f.read()

    complete_data = []
    for log in txt.split('\n'):
        try:
            data = dict()
            log = log.strip()
            data['text'] = log
            entities = []
            res = re.findall(r'\[.*?\]',log)
            entities.append((log.index(res[0][1:-1]), log.index(res[0][1:-1])+len(res[0][1:-1]), "TIMESTAMP"))
            res = re.findall(r'\".*?\"',log)
            entities.append((log.index(res[0][1:-1]), log.index(res[0][1:-1])+len(res[0][1:-1]), "HTTP_REQUEST"))
            entities.append((log.index(log.split(' "')[-1]), log.index(log.split(' "')[-1])+len(log.split(' "')[-1]), "HTTP_RESPONSE"))
            ips = ip_reg.findall(log)
            if len(ips)>0:
                entities.append((log.index(ips[0]), log.index(ips[0])+len(ips[0]), 'IP_ADDRESS'))
            data['entities'] = entities
            complete_data.append(data)
        except:
            continue
    return complete_data

def mongodb_log(filename):
    with open(filename, 'r') as f:
        txt = f.read()

    txt = txt.replace("true", "True")
    txt = txt.replace("false", "False")
    x = txt.split('\n')
    complete_data = []
    for log in x:
        try:
            data = dict()
            log = log.strip()
            entities = []
            entities.append((18, 46,"TIMESTAMP"))
            entities.append((log.index("msg")+7,log.index("msg")+7+log[log.index("msg")+7:].index("\""),"MESSAGE"))
            ips = ip_reg.findall(log)
            if len(ips)>0:
                entities.append((log.index(ips[0]), log.index(ips[0])+len(ips[0]), 'IP_ADDRESS'))
            data['text'] = log
            data['entities'] = entities
            complete_data.append(data)
        except:
            continue
    return complete_data


def concat_json(file_list):
    data = []
    for filename in file_list:
        with open(filename, 'r') as f:
            file_data = json.load(f)
            if len(file_data)>2000:
                data += file_data[:2000]
            else:
                data += file_data
    return data

with open('./data/clean_data/complete.json', 'w') as f:
    json.dump(concat_json(['./data/clean_data/mongodb2.json', './data/clean_data/weblog.json', './data/clean_data/mysql.json', './data/clean_data/apache.json']), f)
# with open('./data/clean_data/mongodb2.json', 'w') as f:
#     json.dump(mongodb_log('./data/mongod.log'), f)