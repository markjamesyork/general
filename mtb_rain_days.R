## calculates percent of days that mtb trails will be closed in minneapolis

# set number of "days" to run simulation
num_trials = 100000

#initiate vector for trial, each position is a day
trial = vector(length = num_trials)

# creates trial data
for (i in 1:num_trials){
  prob_rain <- rbeta(1,1,3)  # draws probability of rain from beta dist
  trial[i]<- rbinom(1, 1, prob_rain) # binarizes rain/not for that trial day
}

# inits closed day vector with assumption that raining day will be closed
closed_days = trial

# allows for drying out time of trail
for (day in 1:length(trial)){
  if (trial[day] == 1){  # if it rained
    num_days_closed = rnorm(n = 1, mean = 1, sd = 0.3) # get drying time from dist (in reality depends on amount of rain)
    closed_days[day+round(num_days_closed)] = 1 # adds trail closes due to drying time (rounds to nearest day..)
  }
}

# prints percent of days that MTB trails are closed
pct_closed = sum(closed_days)/length(closed_days)