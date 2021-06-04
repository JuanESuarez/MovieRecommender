##########################################################
# Movielens Project
# Author: Juan Eloy Suarez
##########################################################

##########################################################
# Initialize environment
##########################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(tidytext)) install.packages("tidytext", repos = "http://cran.us.r-project.org")
if(!require(textdata)) install.packages("textdata", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(data.table)

library(dslabs)
library(lubridate)
library(tidytext) # For sentiment analysis
library(ggrepel)
library(ggplot2)
library(ggthemes)
library(gridExtra)

options(digits=5)

Sys.setlocale("LC_TIME", "english")

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


# Separating Year form title to get a new prescriptor
movies <- movies %>% mutate (year=as.numeric(str_match(title, "(^.*)\\s[(](\\d{4})[)]$")[,3]),
                             title=str_match(title, "(^.*)\\s[(](\\d{4})[)]$")[,2])

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]

temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, removed, movielens)


##########################################################
# Generic functions, parameters
##########################################################

# Function to calculate RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}
# Average of all ratings
mu <- mean(edx$rating)


##########################################################
# Exploratory data analysis
##########################################################
# ===============
# Movies per user
nUsers_nMovies <- edx %>% summarize(n_users = n_distinct(userId), n_movies = n_distinct(movieId))
nUsers_nMovies[1] # Total numbers of users
nUsers_nMovies[2] # Total numbers of movies

# ===================================
# SEGMENT USER: Users and movies exploratory analysis
# ===============
# USER effect
byUser_ratings <- edx %>%
  group_by(userId) %>%
  summarize(rating = mean(rating), b_u_resid = rating - mu)
# Visualize histogram
byUser_ratings %>%
  filter(n()>=100) %>%
  ggplot(aes(rating)) +
  geom_histogram(bins = 30, color = "black")
# Normality test by user
qqnorm(byUser_ratings$rating, pch = 1, frame = FALSE)
qqline(byUser_ratings$rating, col = "steelblue", lwd = 2)
# Visualize distribution of all ratings by rating value
edx %>% ggplot(aes(x = rating)) +
  geom_histogram(bins = 30, color = "black") +
  ggtitle("Distribution of all ratings (in millions)") +
  ylab(element_blank()) + 
  xlab("Rating") + xlim(0,NA) + 
  scale_y_continuous(labels=function(x)x/1000000) +# in millions
  theme(legend.position = "none",
        panel.grid = element_blank(),
        axis.text.x = element_text(color="blue", size=14, face="bold"),
        axis.text.y = element_text(color="blue", size=14, face="bold"))

# ===================================
# SEGMENT MOVIE: We focus on prescriptors related directly with the movie
# ===============
# MOVIE effect
byMovie_ratings <- edx %>% 
  group_by(movieId) %>%
  summarize(rating=mean(rating), Residual = mean(rating - mu))
head(byMovie_ratings)
mu_byMovie <- mean(byMovie_ratings$rating)  # Average of ratings movie to movie
mu_byMovie
# Explore movie
qplot(Residual, geom ="histogram", bins = 10, data = byMovie_ratings, color = I("black"))
# ===============
# YEAR bias
byYear_ratings <- edx %>%
  group_by(Year = year) %>%
  summarize(Residual = mean(rating) - mu, n = n(), Percent=100*Residual/mu) %>%
  filter(!is.na(year)) %>%
  arrange(desc(abs(Residual)))
byYear_ratings
# Visualize diff. percents over the movie's years
byYear_ratings %>% filter(abs(Residual)<1.5) %>%
  ggplot(aes(Year, Percent, color=n)) +
  geom_point(size = 3) +
  geom_hline(yintercept=0, linetype="dashed", color = "orange", size=2) +
  geom_smooth(method = "lm")
# ===============
# GENRE (individualized) bias
tmp <- setNames(strsplit(as.character(edx$genres), split = "[|]"), edx$rating)
indiv_genres <- data.frame(genre = unlist(tmp), rating = as.numeric(as.character(rep(names(tmp), sapply(tmp, length)))), row.names = NULL)
byIndivGenre_ratings <- indiv_genres %>%
  group_by(genre) %>%
  summarize(b_g = mean(rating) - mu, n=n(), Percent=100*b_g/mu) %>%
  mutate(genre=reorder(genre,b_g,FUN=median)) %>%
  arrange(desc(abs(Percent)))
rm(tmp, indiv_genres)
byIndivGenre_ratings
# We are visualizing movies based on their genre
byIndivGenre_ratings %>%
  ggplot(aes(genre, Percent, size=n)) +
  geom_point(color="blue") +
  geom_hline(yintercept=0, linetype="dashed", color = "orange", size=2) +
  geom_label_repel(aes(label = genre),
                   label.size = NA,
                   fill = "transparent",
                   box.padding   = 0.65,
                   point.padding = 0.5,
                   segment.color = 'grey50') +
  ggtitle("Ratings of movies of a genre") +
  xlab(element_blank()) +
  ylab("% diff") +
  theme(legend.position = "none",
        panel.grid = element_blank(),
        axis.text.x = element_blank(),
        axis.text.y = element_text(color="blue", size=14, face="bold"))
# ===============
# Sentiments based on TITLE bias
# Extracting words contained in each movie's title
pattern <- "([^A-Za-z\\d#@']|'(?![A-Za-z\\d#@]))"
movie_words <- edx %>%
  group_by(movieId) %>%
  summarize(title = first(title), rating=mean(rating)) %>%
  mutate(title = str_replace_all(title, "[(+)]", ""))  %>%
  unnest_tokens(word, title, token = "regex", pattern = pattern) %>%
  select(movieId, word, rating)
# Specific lexicons to try
# Identifies EMOTIONS (NRC) : {anger, anticipation, disgust, fear, joy, negative, positive, sadness, surprise, trust}:
myLexicon_nrc <- get_sentiments("nrc") %>% select(word, sentiment)
# Identifies LEVELS (BING): {positive, negative}:
myLexicon_bing <- get_sentiments("bing") %>% select(word, sentiment)
# Calculate ratings of moving containing a specific sentiment based on NRC Lexicon
bySentiment_ratings_nrc <- movie_words %>%
  inner_join(myLexicon_nrc, by = "word") %>%
  group_by(sentiment) %>%
  summarize(b_sr = mean(rating)-mu_byMovie, n=n()) %>%
  mutate(sentiment=reorder(sentiment,b_sr,FUN=median), Percent=100*b_sr/mu_byMovie)
# Calculate ratings of moving containing a specific sentiment based on BING Lexicon
bySentiment_ratings_bing <- movie_words %>%
  inner_join(myLexicon_bing, by = "word") %>%
  group_by(sentiment) %>%
  summarize(Residual = mean(rating)-mu_byMovie, n=n()) %>%
  mutate(sentiment=reorder(sentiment,Residual,FUN=median), Percent=100*Residual/mu_byMovie)

# Visualize NRC sentiments ratings bias
chartbySentiment_ratings_nrc <- bySentiment_ratings_nrc %>%
  ggplot(aes(sentiment, Percent, size=n)) +
  geom_point(color = "blue") +
  geom_hline(yintercept=0, linetype="dashed", color = "orange", size=2) +
  geom_label_repel(aes(label = sentiment),
                   label.size = NA,
                   fill = "transparent",
                   box.padding   = 0.65,
                   point.padding = 0.5,
                   segment.color = 'grey50') +
  ggtitle("Rating of movies inspiring a sentiment in the title") +
  xlab(element_blank()) +
  ylab("% diff.") +
  theme(legend.position = "none",
        panel.grid = element_blank(),
        axis.text.x = element_blank(),
        axis.text.y = element_text(color="blue", size=14, face="bold"))
chartbySentiment_ratings_nrc

# Show bias based on positive/negative summary
bySentiment_ratings_bing

# To apply to each movie, we summarize per movie positive or negative sentiments detected
byMovie_SentimentEffect <-
  (movie_words %>% inner_join(myLexicon_bing, by = "word")) %>%
  left_join(bySentiment_ratings_bing, id = "sentiment") %>%
  mutate(weight=if_else(sentiment=="negative", -1, 1)) %>%
  group_by(movieId) %>%
  summarize(weightSentiment = sum(weight))
head(byMovie_SentimentEffect)
# Range from very negative to very positive sentiments per movie
range(byMovie_SentimentEffect$weightSentiment)

# ===================================
# SEGMENT RATING-MOMENT: Circumstances at the rating event (week day, month, time since movie's year)
# ===============
# WEEKDAY bias
byWeekday_ratings <- edx %>%
  mutate(WeekDay = factor(weekdays(as_datetime(edx$timestamp)))) %>%
  group_by(WeekDay) %>%
  summarize(b_wk = mean(rating) - mu, n=n(), Percent=round(100*b_wk/mu, digits = 2)) %>% 
  arrange(Percent) %>% 
  select(WeekDay, n, Percent)
byWeekday_ratings
# ===============
# MONTH bias
byMonth_ratings <- edx %>%
  mutate(Month = month(as_datetime(timestamp), label = TRUE, abbr = FALSE)) %>%
  group_by(Month) %>%
  summarize(b_m = mean(rating) - mu, n=n(), Percent=round(100*b_m/mu, digits = 2)) %>% 
  arrange(Percent) %>% 
  select(Month, n, Percent)
byMonth_ratings
# Explore month
byMonth_ratings %>%
  ggplot(aes(Month, Percent)) +
  geom_point(color = "blue", size = 3) +
  geom_hline(yintercept=0, linetype="dashed", color = "orange", size=2) +
  geom_label_repel(aes(label = Month),
                   label.size = NA,
                   fill = "transparent",
                   box.padding   = 0.65,
                   point.padding = 0.5,
                   segment.color = 'grey50') +
  ggtitle("Ratings diff. to global average per month when done") +
  theme(legend.position = "none",
        panel.grid = element_blank(), axis.title.x = element_blank(),
        axis.text.x = element_blank())
# We check prevalence is approximately homogeneous for Weekday and Month. This will prevent us from the need of regularization
# WeekDay distribution piechart
pie(byWeekday_ratings$n,
    labels = paste(byWeekday_ratings$WeekDay, sep = " ", round(byWeekday_ratings$n/10^6, digits = 2), "M"), 
    col = rainbow(length(byWeekday_ratings$Percent)), 
    main = "Volume of ratings by day of the week")
# Month distribution piechart
pie(byMonth_ratings$n,
    labels = paste(byMonth_ratings$Month, sep = " ", round(byMonth_ratings$n/10^6, digits = 2), "M"), 
    col = rainbow(length(byMonth_ratings$Percent)), 
    main = "Volume of ratings by month")
# ===============
# Explore AGE bias
minRecords <- nrow(edx) / (5*100) # To avoid non-significant low prevalence cases, let's exclude ages with approx less than 5 times a theoretical equally shared distribution per age in years (considering 0 to 100 possible years of age)
byAgeRating_ratings <-  edx %>%
  mutate(YearRating = year(as_datetime(timestamp)), AgeAtRating = YearRating - year) %>%
  group_by(AgeAtRating) %>%
  summarize(b_ag = mean(rating) - mu, n=n(), Percent=100*b_ag/mu) %>%
  filter(!is.na(AgeAtRating) & (AgeAtRating >= 0) & (n > minRecords)) %>%
  arrange(desc(abs(Percent))) %>% select(AgeAtRating, n, Percent)
byAgeRating_ratings
# Visualize residuals over the movie's years
chart_byAge <- byAgeRating_ratings %>% 
  ggplot(aes(AgeAtRating, Percent, color=n)) +
  geom_point(size = 3) +
  geom_hline(yintercept=0, linetype="dashed", color = "orange", size=2) +
  geom_smooth(method = "lm") + 
  scale_x_reverse() +
  ggtitle("Years between movie release and the moment user rates")
chart_byAge


##########################################################
# Train and predict
##########################################################

mu <- mean(edx$rating)  # We use mean 'mu' as starting point to advance in effects removal
# ==========
# Base prediction
mean_RMSE <- RMSE(validation$rating, mu)  # Calculate RMSE
mean_results <- data_frame(Method="Mean as base reference", RMSE = mean_RMSE) # Store value
mean_results %>% knitr::kable(digits = 4) # Show results

# ==============================================================================
# ==============================================================================
# Regularization (lambda estimate)
# Optimal lambda: Optimize lambda by minimizing RMSE, but using only training set (not test!). We use a sub-partition.

# ==============
# Let's separate our test dataset to a train+test to evaluate some parameters only against training
# ==============
devReduction <- 0.02 # Percentage of original data we extract for development
set.seed(1, sample.kind="Rounding")
reduced_index <- createDataPartition(y = edx$rating, times = 1, p = devReduction, list = FALSE)
reduced_edx <- edx[reduced_index,]

devSplit <- 0.2 # partition train::test
set.seed(1, sample.kind="Rounding")
reduced2_index <- createDataPartition(y = reduced_edx$rating, times = 1, p = devSplit, list = FALSE)

edx_r2 <- reduced_edx[-reduced2_index,]
temp <- reduced_edx[reduced2_index,]

# Make sure userId and movieId in validation set are also in edx set
validation_r2 <- temp %>%
  semi_join(edx_r2, by = "movieId") %>%
  semi_join(edx_r2, by = "userId")
# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation_r2)
edx_r2 <- rbind(edx_r2, removed)
rm(reduced_edx, reduced_index, reduced2_index, temp, removed, devReduction, devSplit)
# ==============
# ==============

lambdas <- seq(1, 10, 0.85)

lRMSEs <- sapply(lambdas, function(l){
  b_i <- edx_r2 %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx_r2 %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_g <- edx_r2 %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
  
  b_y <- edx_r2 %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_g, by="genres") %>%
    group_by(year) %>%
    summarize(b_y = sum(rating - b_i - b_u - b_g - mu)/(n()+l))
  
  predicted_ratings <- validation_r2 %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by="genres") %>%
    left_join(b_y, by = "year") %>%
    mutate(pred = mu + b_i + b_u + b_g + b_y)
  
  return(RMSE(validation_r2$rating, predicted_ratings$pred))
})

chartLambda <- ggplot() + aes(lambdas, lRMSEs) + geom_point() +
  xlab('Lambda') + ylab("RMSE") + ggtitle("Lambda Tuning")
chartLambda
lambda <- lambdas[which.min(lRMSEs)]  # Now we know the optimal lambda value for regularization
lambda

# End of optimal lambda estimation to allow regularization
# ==============================================================================
# ==============================================================================

# ==========
# Training
b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

b_u <- edx %>%
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

b_g <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+lambda))

b_y <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_g, by="genres") %>%
  group_by(year) %>%
  summarize(b_y = sum(rating - b_i - b_u - b_g - mu)/(n()+lambda))

b_sr <- (left_join(edx, byMovie_SentimentEffect, by="movieId")) %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_g, by="genres") %>%
  left_join(b_y, by="year") %>%
  group_by(movieId) %>%
  summarize(b_sr = sum(rating - b_i - b_u - b_g - b_y - mu)/(n()+lambda))

# ==========
# Prediction
predicted_ratings <- validation %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_y, by = "year") %>%
  mutate(pred = mu + b_i + b_u + b_g + b_y)

# ==========
# Calculate and store RMSE
mean_umgR_RMSE <- RMSE(validation$rating, predicted_ratings$pred)
mean_results <- bind_rows(mean_results, data_frame(Method="Regularized Mov+Usr+Gnr+Yr", RMSE = mean_umgR_RMSE))
mean_results %>% knitr::kable(digits = 4)




# ========================================================
# Phase 2: Starting on unbiased values, we'll incorporate features related with the rating moment in time. Actually, we are ensembling into a second model

# Adding predicted rating (in Phase 1) and calculated time-related prescriptors (using timestamp as input)
edx_ph2 <- edx %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_y, by = "year") %>%
  mutate(WeekDay = factor(weekdays(as_datetime(edx$timestamp))),
         Month = month(as_datetime(timestamp)),
         AgeAtRating = year(as_datetime(timestamp)) - year,
         unbiasedRating = mu + b_i + b_u + b_g + b_y)
edx_ph2
# Adding also to test set
validation_ph2 <- predicted_ratings %>%
  mutate(WeekDay = factor(weekdays(as_datetime(timestamp))),
         Month = month(as_datetime(timestamp)),
         AgeAtRating = year(as_datetime(timestamp)) - year)
validation_ph2


# We see Regularization will not drive us to signigicant changes: biases are so small and distribution of n() per category so homogeneous for new prescriptors (WeekDay, Month, AgeAtRating) that it is not worthy to regularize. Instead, we´ll use lambda2 as zero, though including parameter in the code for further developments if applicable.
lambda2 <- 0

# Training Phase 2 model
b_w <- edx_ph2 %>%
  group_by(WeekDay) %>%
  summarize(b_w = sum(rating - unbiasedRating)/(n()+lambda2))

b_m <- edx_ph2 %>%
  left_join(b_w, by="WeekDay") %>%
  group_by(Month) %>%
  summarize(b_m = sum(rating - b_w - unbiasedRating)/(n()+lambda2))

b_a <- edx_ph2 %>%
  left_join(b_w, by="WeekDay") %>%
  left_join(b_m, by="Month") %>%
  group_by(AgeAtRating) %>%
  summarize(b_a = sum(rating - b_w - b_m - unbiasedRating)/(n()+lambda2))

# Prediction Phase 2 model
# We add to initial predictions rating-time information
validation_ph2 <- validation_ph2 %>%
  left_join(b_w, by = "WeekDay") %>%
  left_join(b_m, by = "Month") %>%
  left_join(b_a, by = "AgeAtRating") %>% mutate(pred = pred + b_w + b_m + b_a)

# Calculate and store RMSE
mean_rMUGYT_RMSE <- RMSE(validation$rating, validation_ph2$pred)
mean_results <- bind_rows(mean_results, data_frame(Method="Regularized rMUGY + TIME", RMSE = mean_rMUGYT_RMSE))
mean_results %>% knitr::kable(digits = 4)



