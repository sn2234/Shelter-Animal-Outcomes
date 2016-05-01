library(caret)

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

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

trainNorm <- as.data.frame(lapply(train, normalize))
trainNorm$AnimalID <- train$AnimalID
trainNorm$OutcomeType <- train$OutcomeType
trainNorm$OutcomeSubtype <- train$OutcomeSubtype

set.seed(12345)
inTrain = createDataPartition(trainNorm$AnimalID, p = 2/3, list = FALSE)

write.csv(trainNorm[inTrain, ], file = "processed_train.csv", row.names = FALSE)
write.csv(trainNorm[-inTrain, ], file = "processed_cv.csv", row.names = FALSE)
