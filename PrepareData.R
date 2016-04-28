library(caret)

# Load training samples
trainRaw <- read.csv("train.csv")

train <- data.frame(
  AnimalID =as.numeric(trainRaw$AnimalID),
  Name = as.numeric(trainRaw$Name),
  DateTime = as.numeric(strptime(trainRaw$DateTime, "%Y-%m-%d %H:%M:%S")),
  OutcomeType = as.numeric(trainRaw$OutcomeType),
  OutcomeSubtype = as.numeric(trainRaw$OutcomeSubtype),
  AnimalType = as.numeric(trainRaw$AnimalType),
  SexuponOutcome = as.numeric(trainRaw$SexuponOutcome),
  AgeuponOutcome = as.numeric(trainRaw$AgeuponOutcome),
  Breed = as.numeric(trainRaw$Breed),
  Color = as.numeric(trainRaw$Color)
)

set.seed(12345)
inTrain = createDataPartition(train$AnimalID, p = 2/3, list = FALSE)

write.csv(train[inTrain, ], file = "processed_train.csv", row.names = FALSE)
write.csv(train[-inTrain, ], file = "processed_cv.csv", row.names = FALSE)
