// Predict the Sale Price of the House in Kaggle//


// Import LinearRegression
import org.apache.spark.ml.regression.LinearRegression

// Optional: Use the following code below to set the Error reporting
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)


// Import SparkSession
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate

//Import DataFrame Test and Train

val traindata = spark.read.option("header", "true").option("inferschema", "true").format("csv").load("train-pro.csv")

val testdata = spark.read.option("header", "true").option("inferschema", "true").format("csv").load("test-pro.csv")

// Print the Schema of the DataFrame
traindata.printSchema()
testdata.printSchema()

///////////////////////
/// Display Data /////
/////////////////////
val colnames = traindata.columns
val firstrow = traindata.head(1)(0)
println("\n")
println("Example Data Row")
for(ind <- Range(1,colnames.length)){
  println(colnames(ind))
  println(firstrow(ind))
  println("\n")
}

////////////////////////////////////////////////////
//// Setting Up DataFrame for Machine Learning ////
//////////////////////////////////////////////////

// Grab only the columns we want
val traindataall = (traindata.select(traindata("SalePrice").as("label"),$"MSZoning",$"Street",$"LotConfig",$"LandSlope",
                    $"Neighborhood",$"Condition1",$"Condition2",$"BldgType",$"HouseStyle",$"OverallQual",$"OverallCond",
                    $"YearBuilt",$"YearRemodAdd",$"MasVnrArea",$"ExterQual",$"ExterCond",$"Foundation",$"BsmtQual",$"BsmtFinType2",
                    $"BsmtFinSF2",$"BsmtUnfSF",$"Heating",$"HeatingQC",$"CentralAir",$"X1stFlrSF",$"X2ndFlrSF",$"BedroomAbvGr",
                    $"TotRmsAbvGrd",$"Fireplaces",$"GarageArea",$"GarageCond",$"WoodDeckSF",$"ScreenPorch",$"Fence",$"SaleCondition")
                   )

// Removed below double data types due to error -> java.lang.IllegalArgumentException: Data type StringType is not supported.
//,$"total_sq_footage",$"total_baths",$"logBSMTFinSF1",$"logTotalBSMTSF",$"logGrLivArea"

//To drop NA values
val traindatareg = traindataall.na.drop()

// A few things we need to do before Spark can accept the data!
// We need to deal with the Categorical columns


// Import VectorAssembler and Vectors
import org.apache.spark.ml.feature.{VectorIndexer, StringIndexer,VectorAssembler,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

// Deal with Categorical Columns

traindatareg.printSchema()
val MSZoningIndexer = new StringIndexer().setInputCol("MSZoning").setOutputCol("MSZoningIndex")
val StreetIndexer = new StringIndexer().setInputCol("Street").setOutputCol("StreetIndex")
val LotConfigIndexer = new StringIndexer().setInputCol("LotConfig").setOutputCol("LotConfigIndex")
val LandSlopeIndexer = new StringIndexer().setInputCol("LandSlope").setOutputCol("LandSlopeIndex")
val NeighborhoodIndexer = new StringIndexer().setInputCol("Neighborhood").setOutputCol("NeighborhoodIndex")
val Condition1Indexer = new StringIndexer().setInputCol("Condition1").setOutputCol("Condition1Index")
val Condition2Indexer = new StringIndexer().setInputCol("Condition2").setOutputCol("Condition2Index")
val BldgTypeIndexer = new StringIndexer().setInputCol("BldgType").setOutputCol("BldgTypeIndex")
val HouseStyleIndexer = new StringIndexer().setInputCol("HouseStyle").setOutputCol("HouseStyleIndex")
val ExterQualIndexer = new StringIndexer().setInputCol("ExterQual").setOutputCol("ExterQualIndex")
val ExterCondIndexer = new StringIndexer().setInputCol("ExterCond").setOutputCol("ExterCondIndex")
val FoundationIndexer = new StringIndexer().setInputCol("Foundation").setOutputCol("FoundationIndex")
val BsmtQualIndexer = new StringIndexer().setInputCol("BsmtQual").setOutputCol("BsmtQualIndex")
val BsmtFinType2Indexer = new StringIndexer().setInputCol("BsmtFinType2").setOutputCol("BsmtFinType2Index")
val HeatingIndexer = new StringIndexer().setInputCol("Heating").setOutputCol("HeatingIndex")
val HeatingQCIndexer = new StringIndexer().setInputCol("HeatingQC").setOutputCol("HeatingQCIndex")
val CentralAirIndexer = new StringIndexer().setInputCol("CentralAir").setOutputCol("CentralAirIndex")
val FenceIndexer = new StringIndexer().setInputCol("Fence").setOutputCol("FenceIndex")
val SaleConditionIndexer = new StringIndexer().setInputCol("SaleCondition").setOutputCol("SaleConditionIndex")
val GarageCondIndexer = new StringIndexer().setInputCol("GarageCond").setOutputCol("GarageCondIndex")


//Encode the Data

val MSZoningEncoder = new OneHotEncoder().setInputCol("MSZoningIndex").setOutputCol("MSZoningVec")
val StreetEncoder = new OneHotEncoder().setInputCol("StreetIndex").setOutputCol("StreetVec")
val LotConfigEncoder = new OneHotEncoder().setInputCol("LotConfigIndex").setOutputCol("LotConfigVec")
val LandSlopeEncoder = new OneHotEncoder().setInputCol("LandSlopeIndex").setOutputCol("LandSlopeVec")
val NeighborhoodEncoder = new OneHotEncoder().setInputCol("NeighborhoodIndex").setOutputCol("NeighborhoodVec")
val Condition1Encoder = new OneHotEncoder().setInputCol("Condition1Index").setOutputCol("Condition1Vec")
val Condition2Encoder = new OneHotEncoder().setInputCol("Condition2Index").setOutputCol("Condition2Vec")
val BldgTypeEncoder = new OneHotEncoder().setInputCol("BldgTypeIndex").setOutputCol("BldgTypeVec")
val HouseStyleEncoder = new OneHotEncoder().setInputCol("HouseStyleIndex").setOutputCol("HouseStyleVec")
val ExterQualEncoder = new OneHotEncoder().setInputCol("ExterQualIndex").setOutputCol("ExterQualVec")
val ExterCondEncoder = new OneHotEncoder().setInputCol("ExterCondIndex").setOutputCol("ExterCondVec")
val FoundationEncoder = new OneHotEncoder().setInputCol("FoundationIndex").setOutputCol("FoundationVec")
val BsmtQualEncoder = new OneHotEncoder().setInputCol("BsmtQualIndex").setOutputCol("BsmtQualVec")
val BsmtFinType2Encoder = new OneHotEncoder().setInputCol("BsmtFinType2Index").setOutputCol("BsmtFinType2Vec")
val HeatingEncoder = new OneHotEncoder().setInputCol("HeatingIndex").setOutputCol("HeatingVec")
val HeatingQCEncoder = new OneHotEncoder().setInputCol("HeatingQCIndex").setOutputCol("HeatingQCVec")
val CentralAirEncoder = new OneHotEncoder().setInputCol("CentralAirIndex").setOutputCol("CentralAirVec")
val FenceEncoder = new OneHotEncoder().setInputCol("FenceIndex").setOutputCol("FenceVec")
val SaleConditionEncoder = new OneHotEncoder().setInputCol("SaleConditionIndex").setOutputCol("SaleConditionVec")
val GarageCondEncoder = new OneHotEncoder().setInputCol("GarageCondIndex").setOutputCol("GarageCondVec")


// Assemble everything together to be ("label","features") format
val assembler =   (new VectorAssembler().setInputCols(Array("MSZoningVec","StreetVec","LotConfigVec","LandSlopeVec",
                    "NeighborhoodVec","Condition1Vec","Condition2Vec","BldgTypeVec","HouseStyleVec","OverallQual","OverallCond",
                    "YearBuilt","YearRemodAdd","MasVnrArea","ExterQualVec","ExterCondVec","FoundationVec","BsmtQualVec","BsmtFinType2Vec",
                    "BsmtFinSF2","BsmtUnfSF","HeatingVec","HeatingQCVec","CentralAirVec","X1stFlrSF","X2ndFlrSF","BedroomAbvGr",
                    "TotRmsAbvGrd","Fireplaces","GarageArea","GarageCondVec","WoodDeckSF","ScreenPorch","FenceVec","SaleConditionVec"))
                    .setOutputCol("features")
                   )

//,"total_sq_footage","total_baths","logBSMTFinSF1","logTotalBSMTSF","logGrLivArea"
/// Split the Data ////////
val Array(training, test) = traindatareg.randomSplit(Array(0.7, 0.3), seed = 1234)

///////////////////////////////
// Set Up the Pipeline ///////
/////////////////////////////
import org.apache.spark.ml.Pipeline

val lr = new LinearRegression()

val pipeline = (new Pipeline().setStages(Array( MSZoningIndexer,StreetIndexer,LotConfigIndexer,LandSlopeIndexer,NeighborhoodIndexer,
                Condition1Indexer,Condition2Indexer,BldgTypeIndexer,HouseStyleIndexer,ExterQualIndexer,ExterCondIndexer,FoundationIndexer,
                BsmtQualIndexer,BsmtFinType2Indexer,HeatingIndexer,HeatingQCIndexer,CentralAirIndexer,GarageCondIndexer,
                FenceIndexer,SaleConditionIndexer,MSZoningEncoder,StreetEncoder,LotConfigEncoder,LandSlopeEncoder,NeighborhoodEncoder,
                Condition1Encoder,Condition2Encoder,BldgTypeEncoder,HouseStyleEncoder,ExterQualEncoder,ExterCondEncoder,FoundationEncoder,
                BsmtQualEncoder,BsmtFinType2Encoder,HeatingEncoder,HeatingQCEncoder,CentralAirEncoder,
                FenceEncoder,SaleConditionEncoder,GarageCondEncoder ,assembler, lr))
                )

// Fit the pipeline to training documents.

//val model = pipeline.fit(training)

//Fit the Pipeline to traindatareg
val model = pipeline.fit(traindatareg)


// Get Results on Test Set
//val results =  model.transform(test)

// Get Results on Test Set testdata
val results =  model.transform(testdata)

////////////////////////////////////
//// MODEL EVALUATION /////////////
//////////////////////////////////
results.printSchema()

///////////////////////
/// Display Data /////
/////////////////////
//val colnames = results.prediction
//val firstrow = results.head(1)(0)
//println("\n")
//println("Example Results Data Row")
//for(ind <- Range(1,colnames.length)){
//  println(colnames(ind))
//  println(firstrow(ind))
//  println("\n")
//}

//Write csv
//results.write.format("csv").save(/Users/thejas/Desktop/SMU/Study/Udemy/Scala-Spark-ML-Jose/Scala-and-Spark-Bootcamp-master)

//Kaggle Score .18931
PredictionResults.write.csv("/Users/thejas/Desktop/SMU/Study/Udemy/Scala-Spark-ML-Jose/Scala-and-Spark-Bootcamp-master/KaggleProject/pred3.csv")
// Print the coefficients and intercept for linear regression
//println(s"PredictionResults: ${results.label} prediction: ${results.prediction}")

//println(s"PredictionResults: prediction: ${results.prediction}")


// For Metrics and Evaluation------Below is for Classification Model EVALUATION---------
//import org.apache.spark.mllib.evaluation.MulticlassMetrics

// Need to convert to RDD to use this
//val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd

// Instantiate metrics object
//val metrics = new MulticlassMetrics(predictionAndLabels)

// Confusion matrix
//println("Confusion matrix:")
//println(metrics.confusionMatrix)
