'''Important Variables from ANES:

V201510 -> highest level of education, if value >= 4 (college: Y)
V201600 -> sex, 1=M, 2=F
V201029 -> who they voted for, 1=Biden, 2=Trump
V201033 -> who they will vote for, 1=Biden, 2=Trump
V201507x -> age, 80= 80 or older, -9=N/A'''

import pandas as pd
import numpy as np
import random

def laplace_mech(v, sensitivity, epsilon):
    return v + np.random.laplace(loc=0, scale=sensitivity / epsilon)


data = pd.read_csv("anes_timeseries_2020_csv_20220210.csv",usecols=["V201510","V201600","V201029","V201033","V201507x"])

def preprocc(data):

    #transforming highest level of education into a y/n

    data["college"] = np.where(data["V201510"]>=4,1,0)

    #binning ages into: 18-25, 26-34, 35-46, 47-65, 65-79, 80+, and drop NR values

    data = data.drop(data[data['V201507x'] == -9].index)
    data["age_bins"]=pd.cut(x=data["V201507x"],bins=[17,25,34,46,65,79,81],labels=[1,2,3,4,5,6])

    #sex is as it is, 1=M, 2=F, drop NR values

    data = data.drop(data[data['V201600'] == -9].index)
    data = data.rename(columns={"V201600":"sex"})

    #V201029 contains who the person voted for, V201033 contains who they plan to vote for
    #since we are only concerned about modelling a two-way fight, we discard rows where neither of the 2 values are 1/2 (Biden/Trump)

    biden_condlist, biden_cholist = [data["V201029"]==1, data["V201033"]==1], [1,1]
    trump_condlist, trump_cholist = [data["V201029"]==2, data["V201033"]==2], [1,1]
    data["vote_biden"] = np.select(biden_condlist, biden_cholist,default=0)
    data["vote_trump"] = np.select(trump_condlist, trump_cholist,default=0)
    #having obtained voter choices, we map 1 to Biden and 2 to Trump
    voter_choice, vote = [data["vote_biden"]==1,data["vote_trump"]==1], [1,2]
    data["vote"] = np.select(voter_choice,vote,default=0)
    #removing votes not cast for Biden or Trump
    data = data.drop(data[data['vote'] == 0].index)

    #removing excess columns
    data = data.drop(["V201510","V201507x","V201029","V201033","vote_biden","vote_trump"],axis=1)

    return data


new_data = preprocc(data)
print(data.shape)
#working_data = new_data.drop("age_bins",axis=1)
working_data = new_data
print(working_data.shape)

'''male_college = working_data[(working_data["sex"]==1) & (working_data["college"]==1)]    
male_noncollege = working_data[(working_data["sex"]==1) & (working_data["college"]==0)]
female_college = working_data[(working_data["sex"]==2) & (working_data["college"]==1)]
female_noncollege = working_data[(working_data["sex"]==2) & (working_data["college"]==0)]

mc, mnc, fc, fnc = male_college.shape[0], male_noncollege.shape[0], female_college.shape[0], female_noncollege.shape[0]'''

sexvals = [1,2]
colvals = [0,1]
agevals = [1,2,3,4,5,6]
agebins = ["18-25","26-34","35-46","47-65","65-79","80+"]

dp_distr = []

for i in sexvals:
    for j in colvals:
        for k in agevals:
            data_strata = working_data[(working_data["sex"]==i) & (working_data["college"]==j) & (working_data["age_bins"]==k)]
            num_strata = data_strata.shape[0]
            counts = np.array(data_strata.vote.value_counts())
            dp_counts = [laplace_mech(c, 1, 1) for c in counts]
            dp_probs = dp_counts/np.sum(dp_counts)

            dp_distr.append([i,j,agebins[k-1],dp_probs,num_strata])

print(dp_distr)

def generate_equallylikely(n):
    output = []
    count = int(n/(len(sexvals)*len(colvals)*len(agevals)))

    random.shuffle(dp_distr)

    for i in dp_distr[:-1]:
        random_votes_stra = np.random.choice([1,2],count,p=i[3])
        for j in random_votes_stra:
            output.append([i[0],i[1],i[2],j])
    

    for i in [dp_distr[-1]]:
        new_count = n-len(output)
        random_votes_stra = np.random.choice([1,2],new_count,p=i[3])
        for j in random_votes_stra:
            output.append([i[0],i[1],i[2],j])
    
    random.shuffle(output)
    return pd.DataFrame(np.array(output), columns = ["sex","college","age","vote"])


def generate_originalproportions(n):
    output = []

    random.shuffle(dp_distr)
    total = np.array(dp_distr).T[4].sum()
    print(total)

    for i in dp_distr[:-1]:
        count = int(n*i[4]/total)
        print(count,n,i[4],total)
        random_votes_stra = np.random.choice([1,2],count,p=i[3])
        for j in random_votes_stra:
            output.append([i[0],i[1],i[2],j])
    
    for i in [dp_distr[-1]]:
        new_count = n-len(output)
        #print(new_count,n,i[3],total)
        random_votes_stra = np.random.choice([1,2],new_count,p=i[3])
        for j in random_votes_stra:
            output.append([i[0],i[1],i[2],j])

    random.shuffle(output)
    return pd.DataFrame(np.array(output), columns = ["sex","college","age","vote"])

result = generate_originalproportions(50000)
print(result)

'''counts_mc = np.array(male_college.vote.value_counts())
print(counts_mc)
dp_counts_mc = [laplace_mech(c, 1, 0.1) for c in counts_mc]
dp_probs_mc = dp_counts_mc/np.sum(dp_counts_mc)


def gen_random(k):
    random_voters_mc = np.random.choice([1,2],n,dp_probs_mc)'''