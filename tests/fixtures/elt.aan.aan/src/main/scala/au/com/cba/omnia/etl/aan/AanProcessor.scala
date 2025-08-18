package au.com.cba.omnia.etl.aan

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import com.typesafe.config.Config

// Import other classes from this package
import au.com.cba.omnia.etl.aan.AanCreateDd1
import au.com.cba.omnia.etl.aan.AanExport
import au.com.cba.omnia.etl.aan.AanNotification

/**
 * AanProcessor - Main orchestrator for Account Analysis ETL pipeline
 * 
 * This class coordinates the execution of data processing, export, and notification
 * steps in the Account Analysis workflow. It demonstrates internal package imports
 * and method calls between classes.
 */
class AanProcessor(spark: SparkSession, config: Config) extends Serializable {
  
  import spark.implicits._
  
  private val logger = org.slf4j.LoggerFactory.getLogger(this.getClass)
  
  // Initialize pipeline components
  private val dataProcessor = new AanCreateDd1(spark, config)
  private val dataExporter = new AanExport(spark, config)
  private val notificationService = new AanNotification(config)
  
  /**
   * Execute the complete AAN ETL pipeline
   */
  def executePipeline(): Unit = {
    logger.info("Starting AAN ETL pipeline execution")
    
    try {
      // Step 1: Process raw data
      logger.info("Step 1: Processing raw account data")
      dataProcessor.execute()
      
      // Step 2: Export processed data
      logger.info("Step 2: Exporting processed data")
      dataExporter.execute()
      
      // Step 3: Send notifications
      logger.info("Step 3: Sending completion notifications")
      notificationService.sendCompletionNotification("Pipeline completed successfully")
      
      logger.info("AAN ETL pipeline completed successfully")
      
    } catch {
      case ex: Exception =>
        logger.error(s"Pipeline execution failed: ${ex.getMessage}", ex)
        notificationService.sendErrorNotification(s"Pipeline failed: ${ex.getMessage}")
        throw ex
    }
  }
  
  /**
   * Validate pipeline configuration and dependencies
   */
  def validateConfiguration(): Boolean = {
    logger.info("Validating pipeline configuration")
    
    val validations = Seq(
      validateSparkConfiguration(),
      validateInputPaths(),
      validateOutputPaths(),
      validateDatabaseConnections()
    )
    
    val allValid = validations.forall(identity)
    
    if (allValid) {
      logger.info("Configuration validation passed")
    } else {
      logger.error("Configuration validation failed")
      notificationService.sendErrorNotification("Configuration validation failed")
    }
    
    allValid
  }
  
  private def validateSparkConfiguration(): Boolean = {
    try {
      val sparkConfig = spark.conf.getAll
      val requiredConfigs = Seq(
        "spark.sql.adaptive.enabled",
        "spark.sql.adaptive.coalescePartitions.enabled"
      )
      
      requiredConfigs.forall(sparkConfig.contains)
    } catch {
      case _: Exception => false
    }
  }
  
  private def validateInputPaths(): Boolean = {
    try {
      val inputPath = config.getString("paths.input")
      spark.read.parquet(inputPath).limit(1).count()
      true
    } catch {
      case _: Exception => false
    }
  }
  
  private def validateOutputPaths(): Boolean = {
    try {
      val outputPath = config.getString("paths.output")
      // Check if path is writable
      val testDf = Seq(("test", 1)).toDF("key", "value")
      testDf.write.mode("overwrite").parquet(s"$outputPath/validation_test")
      true
    } catch {
      case _: Exception => false
    }
  }
  
  private def validateDatabaseConnections(): Boolean = {
    try {
      val dbUrl = config.getString("database.url")
      val dbUser = config.getString("database.user") 
      val dbPassword = config.getString("database.password")
      
      // Simple connection test
      val testQuery = "(SELECT 1 as test_connection) as test"
      spark.read
        .format("jdbc")
        .option("url", dbUrl)
        .option("user", dbUser)
        .option("password", dbPassword)
        .option("dbtable", testQuery)
        .load()
        .count()
      
      true
    } catch {
      case _: Exception => false
    }
  }
  
  /**
   * Get pipeline execution metrics
   */
  def getExecutionMetrics(): Map[String, Any] = {
    Map(
      "spark_version" -> spark.version,
      "application_id" -> spark.sparkContext.applicationId,
      "executors" -> spark.sparkContext.getExecutorInfos.length,
      "default_parallelism" -> spark.sparkContext.defaultParallelism,
      "total_cores" -> spark.sparkContext.getExecutorInfos.map(_.totalCores).sum
    )
  }
}

/**
 * Companion object for standalone execution
 */
object AanProcessor {
  
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("AAN-Pipeline-Processor")
      .config("spark.sql.adaptive.enabled", "true")
      .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
      .config("spark.sql.adaptive.skewJoin.enabled", "true")
      .getOrCreate()
    
    try {
      // Load configuration
      val config = com.typesafe.config.ConfigFactory.load()
      
      // Create and execute processor
      val processor = new AanProcessor(spark, config)
      
      // Validate before execution
      if (processor.validateConfiguration()) {
        processor.executePipeline()
        
        // Log execution metrics
        val metrics = processor.getExecutionMetrics()
        println(s"Pipeline execution metrics: $metrics")
      } else {
        throw new RuntimeException("Configuration validation failed")
      }
      
    } finally {
      spark.stop()
    }
  }
}