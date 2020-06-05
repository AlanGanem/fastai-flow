import sys
sys.path.append("..")


def build_features(
        data,
):

    #create coppy of df
    #data = data.copy()
    # do some features engineering
    #date_col = difflib.get_close_matches('Data do Documento', list(data.columns), n=3, cutoff=0.3)[0]
    #print('Closest date col: {}'.format(date_col))

    #data[date_col] = pd.to_datetime(data[date_col], errors = 'coerce', format = '%d/%m/%Y')

    #data['InterUF'] = data['UFUnd'] != data['UF']
    #descricao_col = difflib.get_close_matches('Descriçãodomaterial', list(data.columns), n=3, cutoff=0.3)[0]
    #print('Closest description col: {}'.format(descricao_col))
    # get first word of material description
    #def get_description(data,index = 0):
    #    try:
    #        return data[index].lower()
    #    except:
    #        return ''

    #data['desc'] = (data[descricao_col]).str.strip(' ')
    #data['desc'] = data['desc'].str.strip(';')
    #data['desc'] = data['desc'].str.strip(';')
    #data['desc'] = data['desc'].str.replace(';',' ')
    #data['desc'] = data['desc'].str.strip(';')
    #data['desc1'] = data['desc'].str.split(' ').apply(partial(get_description, index = 0))
    # reset_index in order to split by date
    #data = data.sort_values(by=date_col,ascending = True)
    #data = data.reset_index(drop=True)

    return data
