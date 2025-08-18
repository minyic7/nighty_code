package au.com.cba.omnia.etl.aan

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import com.typesafe.config.Config

/**
 * AanCreateDd1 - Account Analysis ETL Job for Data Domain 1
 * 
 * This job processes raw account data and creates the first data domain
 * for account analysis, including customer demographics, account balances,
 * and transaction summaries.
 */
class AanCreateDd1(spark: SparkSession, config: Config) extends Serializable {
  
  import spark.implicits._
  
  private val logger = org.slf4j.LoggerFactory.getLogger(this.getClass)
  
  // Configuration parameters
  private val inputPath = config.getString("paths.input")
  private val outputPath = config.getString("paths.output")
  private val batchSize = config.getInt("processing.batch_size")
  
  /**
   * Main execution method for the ETL job
   */
  def execute(): Unit = {
    logger.info("Starting AanCreateDd1 ETL job execution")
    
    try {
      // Read raw account data
      val rawAccountData = readRawAccountData()
      
      // Apply transformations
      val processedData = transformAccountData(rawAccountData)
      
      // Apply data quality checks
      val validatedData = applyDataQualityChecks(processedData)
      
      // Write to output
      writeProcessedData(validatedData)
      
      logger.info("AanCreateDd1 ETL job completed successfully")
      
    } catch {
      case ex: Exception =>
        logger.error(s"AanCreateDd1 ETL job failed: ${ex.getMessage}", ex)
        throw ex
    }
  }
  
  /**
   * Read raw account data from input source
   */
  private def readRawAccountData(): DataFrame = {
    logger.info(s"Reading raw account data from: $inputPath")
    
    spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("multiline", "true")
      .option("escape", "\"")
      .csv(s"$inputPath/accounts/*.csv")
      .filter($"account_id".isNotNull)
  }
  
  /**
   * Transform raw account data into processed format
   */
  private def transformAccountData(rawData: DataFrame): DataFrame = {
    logger.info("Applying account data transformations")
    
    rawData
      .withColumn("processing_date", current_date())
      .withColumn("processing_timestamp", current_timestamp())
      .withColumn("account_balance_category", categorizeBalance($"current_balance"))
      .withColumn("customer_age_group", categorizeAge($"customer_age"))
      .withColumn("account_tenure_months", 
        months_between(current_date(), $"account_open_date"))
      .withColumn("last_transaction_days_ago", 
        datediff(current_date(), $"last_transaction_date"))
      .withColumn("is_high_value_customer", 
        when($"current_balance" > 100000, true).otherwise(false))
      .withColumn("risk_score", calculateRiskScore(
        $"current_balance", 
        $"transaction_count_30d", 
        $"account_tenure_months"
      ))
      .select(
        $"account_id",
        $"customer_id", 
        $"account_type",
        $"current_balance",
        $"account_balance_category",
        $"customer_age",
        $"customer_age_group",
        $"account_tenure_months",
        $"last_transaction_days_ago",
        $"transaction_count_30d",
        $"transaction_amount_30d",
        $"is_high_value_customer",
        $"risk_score",
        $"processing_date",
        $"processing_timestamp"
      )
  }
  
  /**
   * Apply data quality validation checks
   */
  private def applyDataQualityChecks(data: DataFrame): DataFrame = {
    logger.info("Applying data quality checks")
    
    val validData = data
      .filter($"account_id".isNotNull && $"account_id" =!= "")
      .filter($"customer_id".isNotNull && $"customer_id" =!= "")
      .filter($"current_balance".isNotNull && $"current_balance" >= 0)
      .filter($"account_tenure_months".isNotNull && $"account_tenure_months" >= 0)
      .filter($"risk_score".between(0.0, 1.0))
    
    val originalCount = data.count()
    val validCount = validData.count()
    val rejectedCount = originalCount - validCount
    
    logger.info(s"Data quality summary - Original: $originalCount, Valid: $validCount, Rejected: $rejectedCount")
    
    if (rejectedCount > 0) {
      val rejectionRate = rejectedCount.toDouble / originalCount.toDouble * 100
      if (rejectionRate > 5.0) {
        logger.warn(s"High rejection rate detected: ${rejectionRate}%")
      }
    }
    
    validData
  }
  
  /**
   * Write processed data to output destination
   */
  private def writeProcessedData(data: DataFrame): Unit = {
    logger.info(s"Writing processed data to: $outputPath")
    
    data
      .coalesce(10) // Optimize file count
      .write
      .mode("overwrite")
      .option("compression", "snappy")
      .partitionBy("processing_date", "account_balance_category")
      .parquet(s"$outputPath/dd1/accounts_processed")
      
    logger.info("Data write completed successfully")
  }
  
  /**
   * UDF to categorize account balance
   */
  private def categorizeBalance = udf((balance: Double) => {
    balance match {
      case b if b < 1000 => "LOW"
      case b if b < 10000 => "MEDIUM"
      case b if b < 100000 => "HIGH" 
      case _ => "PREMIUM"
    }
  })
  
  /**
   * UDF to categorize customer age
   */
  private def categorizeAge = udf((age: Int) => {
    age match {
      case a if a < 25 => "YOUNG"
      case a if a < 35 => "MILLENNIAL"
      case a if a < 50 => "MIDDLE_AGED"
      case a if a < 65 => "MATURE"
      case _ => "SENIOR"
    }
  })
  
  /**
   * UDF to calculate risk score
   */
  private def calculateRiskScore = udf((balance: Double, txnCount: Int, tenure: Double) => {
    val balanceScore = math.min(balance / 1000000, 1.0) // Normalize to 0-1
    val activityScore = math.min(txnCount / 100.0, 1.0) // Normalize to 0-1  
    val tenureScore = math.min(tenure / 120.0, 1.0) // Normalize to 0-1 (10 years)
    
    // Lower score = higher risk
    val riskScore = 1.0 - ((balanceScore * 0.4) + (activityScore * 0.3) + (tenureScore * 0.3))
    math.max(0.0, math.min(1.0, riskScore))
  })
}

/**
 * Companion object with main method for standalone execution
 */
object AanCreateDd1 {
  
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("AAN-Create-DD1")
      .config("spark.sql.adaptive.enabled", "true")
      .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
      .getOrCreate()
    
    try {
      // Load configuration (simplified for example)
      val config = com.typesafe.config.ConfigFactory.load()
      
      // Execute ETL job
      val job = new AanCreateDd1(spark, config)
      job.execute()
      
    } finally {
      spark.stop()
    }
  }
}