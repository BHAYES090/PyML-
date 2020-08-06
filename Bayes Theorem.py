import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##To discover Bayes Theorem, let's consider the following:

##Suppose a research study claims any homicidal criminal is 85% likely
##to have played violent video games in the United States.

##The math stops there for media and politicians, but let's dig
##deeper into this. Focus on the semantics of the claim: "any homicidal
##criminal is 85% likely to have played violent video games." Is this
##supposed to lead us to believe we should blame violent video games
##for homicidal behavior?

##This is important. Between video gaming and homicidal behavior,
##we need to discern which causes the other in our hypothesis.
##The question we should be asking: is the probability a person
##played violent video games given they are homicidal, the same as
##the probability a person being homicidal given they played violent
##video games? " If you think the answer is yes, prepare to be
##enlightened by Bayes Theorem.

##We will reason this out step by step, hopefully making the idea intuitive.
##But first, let us accept that these are two entirely different probabilities:

####P(gamer|homicidal) = .85
####P(homicidal|gamer) = ?

##We will start by simply applying Bayes Theorem,
##which allows us to flip a conditional probability
##in the reverse direction. We will dissect why the formula
##works later, but first let's see how it applies.

#Bayes Theorem

####P(A|B) = P(B|A) * P(A) / P(B)

p_gamer_given_homicidal = .85
p_homicidal = 17251.0 / 324000000.0

##print(p_homicidal)

##Merging data from various sources, you believe that 19% of
##the population plays violent video games.

##We now have three pieces of information needed to flip
##our conditional probability. We can now convert our probability
##a person has played violent video games given they are homicidal
##to the probability a person is homicidal given they have played violent video games.

##First let's work out the math by hand:

####P(A|B) = P(B|A) * P(A) / P(B) 
####P(Homicidal|Gamer) = P(Gamer|Homicidal) * P(Homicidal) / P(Gamer) 
####P(Homicidal|Gamer) = (.85 * .00005) / .19
####P(Homicidal|Gamer) = 0.0004

p_gamer_given_homicidal = .85
p_gamer = .19
p_homicidal = .00005

p_homicidal_given_gamer = p_gamer_given_homicidal * p_homicidal / p_gamer

print("Probability of homicidal given gamer: {}".format(p_homicidal_given_gamer * (100)))

##When you run this, you will see that we get .0002, or .02%,
##probability that someone is homicidal given they play violent video games.
##Wow! This is a much different number than 85%, which represents the
##probability someone plays violent video games given they are homicidal.

##So what has happened here? What is so special about this Bayes Theorem formula?

dataframe = pd.read_csv(r'C:\Users\bohayes\AppData\Local\Programs\Python\Python38\database.csv',
                        low_memory=False)

Data_Refined0 = dataframe.drop(['Record ID', 'Agency Code', 'Agency Name', 'Agency Type', 'City',
                      'State', 'Month', 'Incident', 'Crime Solved', 'Victim Sex', 'Victim Age',
                      'Victim Race', 'Victim Ethnicity', 'Perpetrator Sex', 'Perpetrator Race',
                      'Perpetrator Ethnicity', 'Relationship', 'Weapon', 'Victim Count', 'Perpetrator Count',
                      'Record Source'], axis=1)

data_for_totals0 = Data_Refined0[Data_Refined0['Year'] >= 2014]
data_for_totals1 = data_for_totals0.drop(['Perpetrator Age', 'Year'], axis=1)

Data_Refined1 = Data_Refined0.rename(columns={'Perpetrator Age': 'Age'})

Data_Refined2 = Data_Refined1[Data_Refined1['Year'] >= 2014]

Data_Refined3 = Data_Refined2[Data_Refined2['Crime Type'] != 'Manslaughter by Negligence']

Data_Refined4 = Data_Refined3[Data_Refined3['Age'] != '0']

Data_Refined5 = Data_Refined4[Data_Refined4['Age'] != ' ']

Data_Refined6 = pd.to_numeric(Data_Refined5['Age'])

Data_Refined7 = pd.to_numeric(Data_Refined2['Year'].count())

pop_of_us = 324000000.0
pop_homicidal = ((pop_of_us) * .004423)
pop_homicidal_gamers = (pop_homicidal * .02)
hom_gamers = (pop_homicidal - pop_homicidal_gamers)

print("Number of Murders Solved Murders in 2014: ", Data_Refined6.count())
print("Number of murders unsolved in 2014: ", Data_Refined7)
print("Percentage of Unsolved Muders in 2014", (Data_Refined6.count() / data_for_totals1.count()*100))
print("Average age of Murderers in 2014: ", Data_Refined6.mean())
print("Percentage of population that are murderers: {}".format(((data_for_totals1.count()) / (pop_of_us) * 100)))
print("Subset of Population that are gamers: {}".format(pop_of_us * .19))
print("Subset of Population that is homicidal: {}".format(pop_homicidal))
print("Subset of Population that is homicidal and gamers: {}".format(pop_homicidal - pop_homicidal_gamers))
print("Percentage of population that is a homicidal gamer: {}".format((hom_gamers / pop_of_us) * 100))

##The above values may be wrong


##################################################################################################################


##Just because we hear "85% of violent criminals are
##likely to play violent video games", it does not mean
##that 85% of people who play video games are violent
##criminals. To get some intuition behind this, it helps to simulate an experiment.

population = 100000.0

p_gamer_given_homicidal = .85
p_gamer = .19
p_homicidal = .00005

gamers_ct = population * p_gamer
homicidal_criminals_ct = population * p_homicidal
gamers_and_homicidal_ct = homicidal_criminals_ct * p_gamer_given_homicidal

print("#Gamers: {}".format(gamers_ct))
print("#Homicidal Criminals: {}".format(homicidal_criminals_ct))
print("#Gamers who are homicidal criminals: {}".format(gamers_and_homicidal_ct))

##What we did is some simple math with a population of
##100,000 people. If 19% of the population plays violent
##video games, then that is 19,000 people. But if .005% of
##the population is homicidal, then that means only 5 are
##homicidal criminals. Of those 5 homicidal criminals, only 4.25 (4 or 5) of
##them are video gamers. See what happened there?

##Are we really going to go after those 19,000 video gamers?
##Even though only 4 or 5 of them are homicidal criminals?

##Notice how something in plain sight gets lost in percentages.
##That percentage of video gamers is relative to one group and can
##be very different relative to another group. What we effectively
##have done is take an uncommon attribute (being homicidal) and
##associated it with a common one (playing video games).

##So how does this trace back to Bayes Theorem? Let's
##modify our code below to calculate the percentage of
##the gamer population that are homicidal:

population = 100000.0

p_gamer_given_homicidal = .85
p_gamer = .19
p_homicidal = .00005

gamers_ct = population * p_gamer
homicidal_criminals_ct = population * p_homicidal
gamers_and_homicidal_ct = homicidal_criminals_ct * p_gamer_given_homicidal

p_homicidal_given_gamer = gamers_and_homicidal_ct / gamers_ct

print("Probability of homicidal given gamer: {}".format(p_homicidal_given_gamer))

##For such simple math, there are so many nuances here!
##Notice the relationship between joint probabilities and
##conditional probabilities, and how they are really just
##based on sub-groups in the population. Let's dive into this next.

##Probably the best way to derive Bayes Theorem from our code
##on the previous page is to expand out every variable in its expression.

##Let's focus on two variables: gamers_and_homicidal_ct
##and p_homicidal_given_gamer. If we trace back the steps,
##we can expand their expressions by replacing the variables
##with their expressions. Here it is re-expressed in the Python code file:

population = 100000.0

p_gamer_given_homicidal = .85
p_gamer = .19
p_homicidal = .00005

gamers_ct = population * p_gamer
homicidal_criminals_ct = population * p_homicidal

# gamers_and_homicidal_ct = homicidal_criminals_ct * p_gamer_given_homicidal
gamers_and_homicidal_ct = population * p_homicidal * p_gamer_given_homicidal

# p_homicidal_given_gamer = gamers_and_homicidal_ct / gamers_ct
p_homicidal_given_gamer = (population * p_homicidal * p_gamer_given_homicidal) / (population * p_gamer)

print("Probability of homicidal given gamer: {}".format(p_homicidal_given_gamer))

##Let's focus on the p_homicidal_given_gamer,
##as this is what we are trying to derive, and it
##contains the components for Bayes Theorem:

#p_homicidal_given_gamer = (population * p_homicidal * p_gamer_given_homicidal) / (population * p_gamer)

##Notice we can completely remove the population variable,
##as it exists in both the numerator and denominator so
##it cancels out. What we are left with is just the probability variables:

#p_homicidal_given_gamer = p_homicidal * p_gamer_given_homicidal / p_gamer

##We now have Bayes Theorem derived as it matches the formula:

#P(A|B) = P(B|A) * P(A) / P(B)
#P(Homicidal|Gamer) = P(Gamer|Homicidal) * P(Homicidal) / P(Gamer)

##So Bayes Theorem is a shortcut to use plain
##probabilities, and not have to do all this
##population work. Here it is simplified in Python,
##removing all population-based variables:

p_gamer_given_homicidal = .85
p_gamer = .19
p_homicidal = .00005

p_homicidal_given_gamer = p_homicidal * p_gamer_given_homicidal / p_gamer

print("Probability of homicidal given gamer: {}".format(p_homicidal_given_gamer))

##Now you can see why Bayes Theorem is arguably
##the most important formula in probability. It
##allows us to make inferences and flip those inferences
##as needed. You will find its application critical in many
##disciplines, including medicine, law, business, economics,
##and anywhere else that touches data and statistics. As a
##defining component of conditional probability, it is a
##critical tool you will use again and again.












