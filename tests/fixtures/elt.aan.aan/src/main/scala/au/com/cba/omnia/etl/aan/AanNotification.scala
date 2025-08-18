package au.com.cba.omnia.etl.aan

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import com.typesafe.config.Config
import scala.util.{Try, Success, Failure}

/**
 * AanNotification - Account Analysis Notification Service
 * 
 * This job processes account analysis results and generates notifications
 * for various stakeholders based on risk levels, thresholds, and business rules.
 */
class AanNotification(spark: SparkSession, config: Config) extends Serializable {
  
  import spark.implicits._
  
  private val logger = org.slf4j.LoggerFactory.getLogger(this.getClass)
  
  // Configuration parameters
  private val inputPath = config.getString("paths.output")
  private val notificationEnabled = config.getBoolean("processing.features.notification_generation")
  
  // Notification thresholds
  private val highRiskThreshold = 0.8
  private val criticalBalanceThreshold = 1000000.0
  private val inactivityDaysThreshold = 90
  
  /**
   * Main execution method for the notification job
   */
  def execute(): Unit = {
    logger.info("Starting AanNotification job execution")
    
    if (!notificationEnabled) {
      logger.info("Notification generation is disabled in configuration")
      return
    }
    
    try {
      // Read processed account data
      val accountData = readAccountData()
      
      // Generate risk-based notifications
      generateRiskNotifications(accountData)
      
      // Generate compliance notifications
      generateComplianceNotifications(accountData)
      
      // Generate operational notifications
      generateOperationalNotifications(accountData)
      
      // Send summary notifications
      sendSummaryNotifications(accountData)
      
      logger.info("AanNotification job completed successfully")
      
    } catch {
      case ex: Exception =>
        logger.error(s"AanNotification job failed: ${ex.getMessage}", ex)
        throw ex
    }
  }
  
  /**
   * Read account analysis data
   */
  private def readAccountData(): DataFrame = {
    logger.info(s"Reading account data from: $inputPath")
    
    spark.read
      .parquet(s"$inputPath/dd1/accounts_processed")
      .filter($"processing_date" === current_date())
  }
  
  /**
   * Generate risk-based notifications
   */
  private def generateRiskNotifications(data: DataFrame): Unit = {
    logger.info("Generating risk-based notifications")
    
    // High risk accounts notification
    val highRiskAccounts = data
      .filter($"risk_score" >= highRiskThreshold)
      .select(
        $"account_id",
        $"customer_id",
        $"risk_score",
        $"current_balance",
        $"last_transaction_days_ago",
        lit("HIGH_RISK_ACCOUNT").as("notification_type"),
        current_timestamp().as("notification_timestamp")
      )
      .withColumn("priority", lit("HIGH"))
      .withColumn("message", 
        concat(
          lit("Account "), $"account_id", 
          lit(" has high risk score: "), $"risk_score"
        )
      )
    
    if (highRiskAccounts.count() > 0) {
      sendNotifications(highRiskAccounts, "risk_management_team")
      
      // Save notification log
      highRiskAccounts.write
        .mode("append")
        .partitionBy("notification_type")
        .parquet(s"$inputPath/notifications/risk_notifications")
    }
    
    // Critical balance accounts
    val criticalBalanceAccounts = data
      .filter($"current_balance" >= criticalBalanceThreshold)
      .filter($"risk_score" >= 0.5) // Medium+ risk with high balance
      .select(
        $"account_id",
        $"customer_id", 
        $"current_balance",
        $"risk_score",
        lit("CRITICAL_BALANCE_HIGH_RISK").as("notification_type"),
        current_timestamp().as("notification_timestamp")
      )
      .withColumn("priority", lit("CRITICAL"))
      .withColumn("message",
        concat(
          lit("High-value account "), $"account_id",
          lit(" (Balance: $"), $"current_balance", 
          lit(") has elevated risk score: "), $"risk_score"
        )
      )
    
    if (criticalBalanceAccounts.count() > 0) {
      sendNotifications(criticalBalanceAccounts, "executive_team")
    }
    
    logger.info("Risk notifications generated")
  }
  
  /**
   * Generate compliance notifications
   */
  private def generateComplianceNotifications(data: DataFrame): Unit = {
    logger.info("Generating compliance notifications")
    
    // Inactive high-value accounts (potential AML concern)
    val inactiveHighValueAccounts = data
      .filter($"current_balance" > 50000)
      .filter($"last_transaction_days_ago" > inactivityDaysThreshold)
      .select(
        $"account_id",
        $"customer_id",
        $"current_balance", 
        $"last_transaction_days_ago",
        lit("INACTIVE_HIGH_VALUE").as("notification_type"),
        current_timestamp().as("notification_timestamp")
      )
      .withColumn("priority", lit("HIGH"))
      .withColumn("message",
        concat(
          lit("High-value account "), $"account_id",
          lit(" inactive for "), $"last_transaction_days_ago", lit(" days")
        )
      )
    
    if (inactiveHighValueAccounts.count() > 0) {
      sendNotifications(inactiveHighValueAccounts, "compliance_team")
      
      // Save compliance notification log
      inactiveHighValueAccounts.write
        .mode("append")
        .partitionBy("notification_type")
        .parquet(s"$inputPath/notifications/compliance_notifications")
    }
    
    logger.info("Compliance notifications generated")
  }
  
  /**
   * Generate operational notifications
   */
  private def generateOperationalNotifications(data: DataFrame): Unit = {
    logger.info("Generating operational notifications")
    
    // Data quality issues
    val dataQualityIssues = identifyDataQualityIssues(data)
    
    if (dataQualityIssues.nonEmpty) {
      val qualityNotification = spark.createDataFrame(
        Seq((
          "SYSTEM",
          "DATA_QUALITY_ALERT", 
          dataQualityIssues.mkString("; "),
          "MEDIUM",
          java.sql.Timestamp.from(java.time.Instant.now())
        )), 
        StructType(Seq(
          StructField("account_id", StringType),
          StructField("notification_type", StringType),
          StructField("message", StringType), 
          StructField("priority", StringType),
          StructField("notification_timestamp", TimestampType)
        ))
      )
      
      sendNotifications(qualityNotification, "data_ops_team")
    }
    
    logger.info("Operational notifications generated")
  }
  
  /**
   * Send summary notifications to stakeholders
   */
  private def sendSummaryNotifications(data: DataFrame): Unit = {
    logger.info("Sending summary notifications")
    
    val totalAccounts = data.count()
    val highRiskCount = data.filter($"risk_score" >= highRiskThreshold).count()
    val highValueCount = data.filter($"is_high_value_customer" === true).count()
    val avgRiskScore = data.agg(avg($"risk_score")).head().getDouble(0)
    
    val summaryMessage = s"""
      |Daily Account Analysis Summary:
      |- Total Accounts Processed: $totalAccounts
      |- High Risk Accounts: $highRiskCount (${(highRiskCount.toDouble/totalAccounts*100).formatted("%.2f")}%)
      |- High Value Customers: $highValueCount (${(highValueCount.toDouble/totalAccounts*100).formatted("%.2f")}%)
      |- Average Risk Score: ${avgRiskScore.formatted("%.3f")}
      |
      |Processing Date: ${java.time.LocalDate.now()}
    """.stripMargin
    
    // Create summary notification record
    val summaryNotification = spark.createDataFrame(
      Seq((
        "DAILY_SUMMARY",
        "SUMMARY_REPORT",
        summaryMessage,
        "LOW", 
        java.sql.Timestamp.from(java.time.Instant.now())
      )),
      StructType(Seq(
        StructField("account_id", StringType),
        StructField("notification_type", StringType), 
        StructField("message", StringType),
        StructField("priority", StringType),
        StructField("notification_timestamp", TimestampType)
      ))
    )
    
    sendNotifications(summaryNotification, "business_stakeholders")
    
    logger.info("Summary notifications sent")
  }
  
  /**
   * Identify data quality issues
   */
  private def identifyDataQualityIssues(data: DataFrame): List[String] = {
    val issues = scala.collection.mutable.ListBuffer[String]()
    
    val totalCount = data.count()
    val nullRiskScoreCount = data.filter($"risk_score".isNull).count()
    val invalidBalanceCount = data.filter($"current_balance" < 0).count()
    
    if (nullRiskScoreCount > 0) {
      issues += s"$nullRiskScoreCount accounts have null risk scores"
    }
    
    if (invalidBalanceCount > 0) {
      issues += s"$invalidBalanceCount accounts have negative balances"
    }
    
    val nullRiskRate = nullRiskScoreCount.toDouble / totalCount * 100
    if (nullRiskRate > 1.0) {
      issues += f"High null risk score rate: $nullRiskRate%.2f%%"
    }
    
    issues.toList
  }
  
  /**
   * Send notifications to specified recipients
   */
  private def sendNotifications(notifications: DataFrame, recipient: String): Unit = {
    logger.info(s"Sending ${notifications.count()} notifications to $recipient")
    
    // In a real implementation, this would integrate with:
    // - Email service (SMTP, SES, etc.)
    // - Slack/Teams webhooks  
    // - SMS service
    // - Incident management systems (PagerDuty, etc.)
    // - Dashboard/monitoring systems
    
    Try {
      // Simulate notification sending
      notifications.collect().foreach { row =>
        val notificationType = row.getAs[String]("notification_type")
        val message = row.getAs[String]("message") 
        val priority = row.getAs[String]("priority")
        
        logger.info(s"[$recipient] [$priority] $notificationType: $message")
        
        // Simulate external API call delay
        Thread.sleep(10)
      }
    } match {
      case Success(_) => 
        logger.info(s"Successfully sent notifications to $recipient")
      case Failure(ex) => 
        logger.error(s"Failed to send notifications to $recipient: ${ex.getMessage}")
    }
  }
}

/**
 * Companion object with main method for standalone execution
 */
object AanNotification {
  
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("AAN-Notification")
      .config("spark.sql.adaptive.enabled", "true")
      .getOrCreate()
    
    try {
      // Load configuration
      val config = com.typesafe.config.ConfigFactory.load()
      
      // Execute notification job
      val job = new AanNotification(spark, config)
      job.execute()
      
    } finally {
      spark.stop()
    }
  }
}