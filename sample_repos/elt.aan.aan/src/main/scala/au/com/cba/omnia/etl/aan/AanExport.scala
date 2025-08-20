package au.com.cba.omnia.etl.aan

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import com.typesafe.config.Config
import java.util.Properties

/**
 * AanExport - Account Analysis Export Job
 * 
 * This job exports processed account analysis data to various downstream systems
 * including databases, file systems, and external APIs for reporting and analytics.
 */
class AanExport(spark: SparkSession, config: Config) extends Serializable {
  
  import spark.implicits._
  
  private val logger = org.slf4j.LoggerFactory.getLogger(this.getClass)
  
  // Configuration parameters
  private val inputPath = config.getString("paths.output")
  private val dbUrl = config.getString("database.url")
  private val dbUser = config.getString("database.user")
  private val dbPassword = config.getString("database.password")
  
  /**
   * Main execution method for the export job
   */
  def execute(): Unit = {
    logger.info("Starting AanExport job execution")
    
    try {
      // Read processed account data
      val processedData = readProcessedData()
      
      // Export to database
      exportToDatabase(processedData)
      
      // Export summary reports
      exportSummaryReports(processedData)
      
      // Export risk analysis
      exportRiskAnalysis(processedData)
      
      logger.info("AanExport job completed successfully")
      
    } catch {
      case ex: Exception =>
        logger.error(s"AanExport job failed: ${ex.getMessage}", ex)
        throw ex
    }
  }
  
  /**
   * Read processed data from data domain 1
   */
  private def readProcessedData(): DataFrame = {
    logger.info(s"Reading processed data from: $inputPath")
    
    spark.read
      .parquet(s"$inputPath/dd1/accounts_processed")
      .filter($"processing_date" === current_date())
  }
  
  /**
   * Export data to database tables
   */
  private def exportToDatabase(data: DataFrame): Unit = {
    logger.info("Exporting data to database")
    
    val dbProperties = new Properties()
    dbProperties.setProperty("user", dbUser)
    dbProperties.setProperty("password", dbPassword)
    dbProperties.setProperty("driver", "org.postgresql.Driver")
    
    // Export main account analysis table
    data.write
      .mode("overwrite")
      .jdbc(dbUrl, "account_analysis_daily", dbProperties)
    
    // Export high-risk accounts table
    val highRiskAccounts = data
      .filter($"risk_score" > 0.7)
      .select($"account_id", $"customer_id", $"risk_score", 
              $"current_balance", $"processing_date")
    
    highRiskAccounts.write
      .mode("overwrite") 
      .jdbc(dbUrl, "high_risk_accounts", dbProperties)
    
    logger.info("Database export completed")
  }
  
  /**
   * Export summary reports
   */
  private def exportSummaryReports(data: DataFrame): Unit = {
    logger.info("Generating summary reports")
    
    // Balance category summary
    val balanceSummary = data
      .groupBy($"account_balance_category")
      .agg(
        count("*").as("account_count"),
        sum($"current_balance").as("total_balance"),
        avg($"current_balance").as("avg_balance"),
        avg($"risk_score").as("avg_risk_score")
      )
      .orderBy($"account_balance_category")
    
    balanceSummary.write
      .mode("overwrite")
      .option("header", "true")
      .csv(s"$inputPath/reports/balance_category_summary")
    
    // Age group summary
    val ageSummary = data
      .groupBy($"customer_age_group")
      .agg(
        count("*").as("customer_count"),
        avg($"current_balance").as("avg_balance"),
        avg($"account_tenure_months").as("avg_tenure_months"),
        avg($"risk_score").as("avg_risk_score")
      )
      .orderBy($"customer_age_group")
    
    ageSummary.write
      .mode("overwrite")
      .option("header", "true")
      .csv(s"$inputPath/reports/age_group_summary")
    
    logger.info("Summary reports exported")
  }
  
  /**
   * Export risk analysis data
   */
  private def exportRiskAnalysis(data: DataFrame): Unit = {
    logger.info("Exporting risk analysis data")
    
    // Risk score distribution
    val riskDistribution = data
      .withColumn("risk_bucket", 
        when($"risk_score" < 0.2, "VERY_LOW")
        .when($"risk_score" < 0.4, "LOW")
        .when($"risk_score" < 0.6, "MEDIUM")
        .when($"risk_score" < 0.8, "HIGH")
        .otherwise("VERY_HIGH")
      )
      .groupBy($"risk_bucket")
      .agg(
        count("*").as("account_count"),
        sum($"current_balance").as("total_balance"),
        avg($"current_balance").as("avg_balance")
      )
      .orderBy($"risk_bucket")
    
    riskDistribution.write
      .mode("overwrite")
      .option("header", "true")
      .csv(s"$inputPath/reports/risk_distribution")
    
    // High-value high-risk accounts for manual review
    val criticalAccounts = data
      .filter($"is_high_value_customer" === true && $"risk_score" > 0.6)
      .select(
        $"account_id",
        $"customer_id", 
        $"current_balance",
        $"risk_score",
        $"last_transaction_days_ago",
        $"account_tenure_months"
      )
      .orderBy($"risk_score".desc)
    
    criticalAccounts.write
      .mode("overwrite")
      .option("header", "true")
      .csv(s"$inputPath/reports/critical_accounts_review")
    
    logger.info("Risk analysis export completed")
  }
}

/**
 * Companion object with main method for standalone execution
 */
object AanExport {
  
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("AAN-Export")
      .config("spark.sql.adaptive.enabled", "true")
      .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
      .getOrCreate()
    
    try {
      // Load configuration
      val config = com.typesafe.config.ConfigFactory.load()
      
      // Execute export job
      val job = new AanExport(spark, config)
      job.execute()
      
    } finally {
      spark.stop()
    }
  }
}