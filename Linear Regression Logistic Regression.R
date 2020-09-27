# load the data and attach the data to the system 

data_set <- read.csv("C:/Users/birthwt2020.csv")
print(data_set)
head(data_set)
attach(data_set)

# check column names

names(data_set)

# initial exploration by scatter plot 

plot(lwt, bwt, main="scatterplot")

# check correlation between two variables
# Can I do a for loop to do all correlations???? Try? 

cor(lwt, bwt)

# fit a linear regression into it for lwt and bwt
LM_model <- lm(bwt ~ lwt)

summary(LM_model)  # look at the statistic summary of the model 

abline(LM_model) # add an regression line into the graph

anova(LM_model) # ANOVA analysis display 

# Review diagnostic plot


plot(LM_model)
#Hit<Return> to see next plot

par(mfrow=c(1,2))
plot(LM_model)

# Multiple linear regression model 
MLM_model <- lm(bwt ~ lwt+age)
summary(MLM_model)







