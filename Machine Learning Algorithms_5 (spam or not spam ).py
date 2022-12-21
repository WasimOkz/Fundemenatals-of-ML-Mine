#!/usr/bin/env python
# coding: utf-8

# # Using naive-bayes algo. to check whether the email is spam or not!
 1) import data file means data
 
 2) convert string data into numeric data
 for this use this method
 
 3) to convert: spam and ham we just to apply
 data.label = data.label.apply(lambda x:1 if x=='spam' else 0) it returns 1 if mail is spam else 0
 
 4) to convert: text data into numbers we have to use Pipeline method for this;
 
 5) from sklearn.model_selection import train_test_split
    x_train,x_test , y_train,y_test = train_test_split(data.text,data.label, test_size = 0.2)
 
 6) from sklearn.naive_bayes import MultinomialNB()
 
 7) from sklearn.feature_extraction.text import CountVectorizer
    v = CountVectorizer()
 
 8) from sklearn.pipeline import Pipeline
 
    pipe_model = Pipeline([ 
                          ('vectorizer', Countvectorizer()),
                          ('nb',MultinomialNB())
                         ])
 
 9)  pipe_model.fit(x_train,y_train)
 
 10) pipe_model.predict(x_test)
 
 11) pipe_model.score(x_test,y_test)
 
# In[52]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv(r"E:\Math\2.Mine_Interest_\Data_Science_Course\Datasets\archive (5)\spam_ham_dataset.csv")
data.head()


# In[3]:


data = data[['label','text']]
data


# In[4]:


data.groupby('label').describe()


# In[5]:


# converting the string into values
# first changins spam and ham using lambda fun


data.label = data.label.apply(lambda x:1 if x=='spam' else 0)
data


# In[6]:


# now converting text column into string using countVictorize!

from sklearn.model_selection import train_test_split

x_train,x_test, y_train,y_test = train_test_split(data.text,data.label, test_size = 0.2)


# In[7]:


from sklearn.feature_extraction.text import CountVectorizer

v = CountVectorizer()


# In[8]:


x_train_count = v.fit_transform(x_train.values)
x_train_count.toarray()


# In[14]:


x_test_count = v.fit_transform(x_test)
x_test_count.toarray()


# In[15]:


from sklearn.naive_bayes import MultinomialNB

mn_model = MultinomialNB()


# In[16]:


mn_model.fit(x_train_count, y_train)


# In[34]:


"""email  = [
     some thing
    
]

email_count = v.fit_transform(email)
mn_model.predict(email_count)

"""

# this will work if you passed some email test it will show you is it spam or not 


# In[42]:


# there is also another method instead of countvectorizer which is pipline one this is simple 
# better to use this one

from sklearn.pipeline import Pipeline

pipe_model = Pipeline( [ ('vectorizer',CountVectorizer()) ,  ('nb' ,MultinomialNB())    ])


# In[43]:


pipe_model.fit(x_train,y_train)


# In[44]:


pipe_model.score(x_test,y_test)


# In[45]:


pipe_model.predict(x_test)


# In[50]:


emails  = [   
""" Subject: july pricing discrepancy : sell to tenaska marketing , deal # 399791
daren -
sell to tenaska marketing
july deal # 399791
tenaska is indicating that the sale for july 2000 should be priced at $ 3 . 56 .
could you confirm our sale price should have been $ 3 . 58 ?
- - - - - - - - - - - - - - - - - - - - - - forwarded by katherine herrera / corp / enron on
10 / 05 / 2000 08 : 03 am - - - - - - - - - - - - - - - - - - - - - - - - - - -
kristen j hanson @ ect
09 / 15 / 2000 09 : 37 am
to : katherine herrera / corp / enron @ enron
cc :
subject : re : cleburne plant
fyi - the purchase invoices have arrived and were given to megan . please feel
free to call darren with any questions .
thanks ,
kris
daren j farmer
09 / 13 / 2000 04 : 01 pm
to : rita wynne / hou / ect @ ect , kristen j hanson / hou / ect @ ect , pat
clynes / corp / enron @ enron , julie meyers / hou / ect @ ect , bob m hall / na / enron @ enron ,
david baumbach / hou / ect @ ect , steve jackson / hou / ect @ ect , mark
mccoy / corp / enron @ enron
cc :
subject : cleburne plant
well , folks , the cleburne deal has started . the deal has not been finalized
through legal . however , we were able to "" get between "" a few deals and make
some money before we finalized it . the cleburne plant was down 7 / 31 - 8 / 7 and
8 / 31 . on these dates , we bought gas from tenaska iv texas partners and sold
to anp and tenaska marketing . because things were not set up , i could not
enter the deals until now . please review the following deals and settle with
the customers .
buy from tenaska iv
july 399763
aug 399773
sell to tenaska marketing
july 399791
aug 399797
sell to anp
aug 399814
the trading desk these are on is "" ena - im cleburne "" .
kris - i have forwarded invoices from tenaska iv to you via intercompany
mail . ( i received both invoices this week )
i am sure there are questions , just give me a call . tanks .
d """
    ,
""" Subject: 06 / 01 ena gas sales on hpl , to d & h gas company
good morning daren :
i received a fax from d & h for 06 / 01 deliveries ( hplc sales to d & h ) , however
there is no sales draft in unify for this sale . please advise if the 360
mmbtu reflected on d & h ' s 06 / 01 allocations , for hpl meter # 428 - deweyville .
tx . a valid hplc sale and if so , who do i contact about getting deal in
sitara and scheduled for hplc , so as to invoice d & h this month !
thanks !
tess ray ( tnray @ aep . com )
aep - houston pipe line co .
1201 louisiana
suite 1200
houston , texas 77002
tel : 832 - 668 - 1248
fax : 832 - 668 - 1123
- - - - - forwarded by tessie n ray / aepin on 07 / 13 / 01 09 : 09 am - - - - -
| | " farmer , daren j . " | |
| | | |
| | | cc : |
| | 06 / 15 / 01 11 : 07 am | subject : |
| | | fw : fw : 05 / 01 ena gas |
| | | sales on hpl , to d & |
| | | h gas company |
> - - - - - original message - - - - -
> from : hernandez , elizabeth l .
> sent : friday , june 15 , 2001 11 : 05 am
> to : farmer , daren j .
> subject : fw : fw : 05 / 01 ena gas sales on hpl , to d & h gas company
>
> deal 70120 has been extended through may 31 , 2001 .
>
>
> - - - - - original message - - - - -
> from : hernandez , elizabeth l .
> sent : friday , june 15 , 2001 9 : 14 am
> to : richardson , stacey
> subject : fw : fw : 05 / 01 ena gas sales on hpl , to d & h gas company
>
> can you help me out with this .
>
> - - - - - original message - - - - -
> from : lambert , karen
> sent : friday , june 15 , 2001 9 : 11 am
> to : hernandez , elizabeth l .
> subject : fw : fw : 05 / 01 ena gas sales on hpl , to d huthmacher , tammie ; aalland @ aep . com
> subject : re : fw : 05 / 01 ena gas sales on hpl , to d & h gas company
>
> the contract ( 96007529 ) should have expired under ena , but because of
> the timing of the expiration , it got " assigned " to aep / hpl ( 96060798 ) .
> therefore , for informational / historical purposes , i extended the term
> under 96060798 thru 5 / 31 / 01 .
>
> someone from linda ' s group will have to extend the term under the
> original contract in order for the deal to be attached .
>
>
>
>
> from : karen lambert / enron @ enronxgate on 06 / 14 / 2001 04 : 34 pm
> to : linda s bryan / enron @ enronxgate , cheryl dudley / hou / ect @ ect
> cc : tammie huthmacher / enron @ enronxgate
>
> subject : fw : 05 / 01 ena gas sales on hpl , to d & h gas company
>
> fyi
>
> - - - - - original message - - - - -
> from : hernandez , elizabeth l .
> sent : thursday , june 14 , 2001 1 : 44 pm
> to : lambert , karen
> cc : farmer , daren j .
> subject : fw : 05 / 01 ena gas sales on hpl , to d & h gas company
> importance : high
>
> unable to add may production as contract needs to be exteneded .
>
> - - - - - original message - - - - -
> from : farmer , daren j .
> sent : thursday , june 14 , 2001 1 : 33 pm
> to : hernandez , elizabeth l .
> subject : fw : 05 / 01 ena gas sales on hpl , to d & h gas company
> importance : high
>
> elizabeth ,
>
> can you extend the contract with d & h under deal 70120 for 1 month to
> cover this volume ?
>
> d
>
> - - - - - original message - - - - -
> from : tnray @ aep . com @ enron
> [ mailto : imceanotes - tnray + 40 aep + 2 ecom + 40 enron @ enron . com ]
> sent : thursday , june 14 , 2001 11 : 44 am
> to : dfarmer @ enron . com
> subject : re : 05 / 01 ena gas sales on hpl , to d & h gas company
> importance : high
>
> daren ,
> per my e - mail to julie and her reply ( see below ) , can you help me with
> this ?
>
> thanks !
> tess
> - - - - - forwarded by tessie n ray / aepin on 06 / 14 / 01 11 : 44 am - - - - -
>
> julie l meyers
> to : tessie n
> ray / aepin @ aepin
> 06 / 14 / 01 10 : 58 cc :
> am subject : re : 05 / 01 ena
> gas sales on hpl , to d & h gas company
> ( document link : tessie n ray )
>
>
>
>
> i did a search and found no deals for may for this counterparty . call
> daren farmer at 713 - 853 - 6905 , and he could probably put the deal in
> for
> you .
>
>
>
>
> ( embedded image moved to file : picl 6827 . pcx ) tessie n ray
> 06 / 13 / 2001 03 : 44 pm
>
>
> to : julie l meyers / hol / aepin @ aepin
> cc : elizabeth . hernandez @ enron . com
>
> subject : 05 / 01 ena gas sales on hpl , to d & h gas company
>
> ( embedded
> image moved ( embedded image moved to file : pico 0491 . pcx )
> to file :
> pico 9961 . pcx )
>
>
>
>
>
>
> julie -
> need your help again . . . .
> d & h gas company , faxed their 05 / 01 volume support for hpl meter 428
> ( 980428 ) , whihc reflects ena gas sales to d & h for 5 / 11 / 01 , 143 mmbtu
> and
> 5 / 31 / 01 for 286 mmbtu for a total of 429 mmbtu for 05 / 01 . he deal last
> month ( 04 / 01 ) , was sa - 70120 . i don ' t have see a sales draft under
> ena or
> hpl for 05 / 01 sales to d & h , nor do i see any 05 / 01 sales deal under
> sa - 70120 .
>
> can you assist me in finding out if there was a sale to d & h in 05 / 01 ,
> and
> if so , what is the deal number and where is sales draft ?
>
> thanks !
> tess
>
>
>
>
>
> - picl 6827 . pcx >
> - pico 9961 . pcx >
> - pico 0491 . pcx >
>
>
>
- picl 6827 . pcx
- pico 9961 . pcx
- pico 0491 . pcx  """
,
    
""" Subject: 1999 hpl unaccounted for
per discussions that hopefully most of you have or will have with your
respective managers , you have been selected to be a part of a team to address
various issues associated with reconciling and clearing 1999 ua 4 . the scope
of the team will be to work as a team , on production months where the ua 4
number per our accounting systems is considerably larger in comparison to
what the pipeline balancing reports has as unaccounted for . this team will
be dedicated fully to this effort and will need to be prepared to contribute
whatever hours are needed to get the desired results . yvette will be
locating a room that we can occupy for all of this week and next week . this
room will be a "" war room "" . everyone is expected to report to the "" war room ""
on a daily basis and should expect to be there unless something extremely
pressing causes you to have to leave or people need to go back to their
offices to retrieve data or information . the expectations are that this
group will work to identify any volume discrepancies , adjustments , etc . and
be able to either correct or quantify those volumetric adjustments by no
later than mar . 24 , 2000 . this team is effective as of tomorrow , mar . 16 ,
2000 at 9 : 00 a . m .
yvette - please try to locate a room by then and notify the above team members
of the location . in the case that you are not able to find a room right
away , please have everyone go to brenda ' s office in eb 3748 at 9 : 00 a . m .
the items on the radar screen to address in resolving volume issues are :
1 . king ranch
2 . gulf plains plant
3 . gulf energy imbalance
4 . hanover treaters
5 . south texas treaters
6 . tejas gas p / l imbalance
7 . channel a / s line unaccounted for
8 . third party imbalances - mops vs . pops """
,
    
"""  "Subject: pictures
streamlined denizen ajar chased
heavens hostesses stolid pinched saturated
staten seventeens juggler abashed
ice guts centrifugal bauxite wader
shyness whirr ukrainian understandingly conditioner
barges entitles vanderpoel
preset wigwam storming alexei
supergroup tab mare
birthright brutalize tolerates depots bubbling
- -
phone : 439 - 120 - 6060
mobile : 590 - 203 - 5805
email : darden . audley @ houston . rr . com





"""
]


# In[51]:


pipe_model.predict(emails)

