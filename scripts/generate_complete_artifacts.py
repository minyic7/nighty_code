"""
Generate complete artifacts including identity cards with LLM summaries.
This script creates a comprehensive set of artifacts for test Scala files.
"""

import json
import os
from pathlib import Path
from datetime import datetime
import shutil

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from nighty_code.core.classifier import FileClassifier
from nighty_code.identity.card_builder import IdentityCardBuilder
from nighty_code.parsers.model.scala_model import ScalaModelExtractor


def create_test_repository():
    """Create a test repository with multiple Scala files."""
    
    test_repo = Path("test_repository")
    test_repo.mkdir(exist_ok=True)
    
    # Create main source directory
    src_dir = test_repo / "src" / "main" / "scala" / "com" / "example"
    src_dir.mkdir(parents=True, exist_ok=True)
    
    # File 1: Main processor
    processor_file = src_dir / "AanProcessor.scala"
    processor_code = """package com.example

import com.example.export.AanExport
import com.example.notification.AanNotification
import com.example.utils.DataValidator
import org.apache.spark.sql.SparkSession

/**
 * Main orchestrator for the AAN ETL pipeline.
 */
class AanProcessor(spark: SparkSession) {
  
  private val validator = new DataValidator()
  private val exporter = new AanExport(spark)
  private val notifier = new AanNotification()
  
  def execute(inputPath: String, outputPath: String): Unit = {
    println(s"Starting AAN processing from $inputPath")
    
    // Validate input
    if (!validator.validatePath(inputPath)) {
      throw new IllegalArgumentException(s"Invalid input path: $inputPath")
    }
    
    // Process data
    val processedData = processData(inputPath)
    
    // Export results
    exporter.export(processedData, outputPath)
    
    // Send notifications
    notifier.sendCompletion("AAN processing completed successfully")
  }
  
  private def processData(path: String): DataFrame = {
    spark.read.parquet(path)
      .filter("status = 'active'")
      .select("id", "name", "value", "timestamp")
  }
}

object AanProcessor {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("AAN Processor")
      .getOrCreate()
    
    val processor = new AanProcessor(spark)
    processor.execute(args(0), args(1))
    
    spark.stop()
  }
}
"""
    processor_file.write_text(processor_code)
    
    # File 2: Export module
    export_dir = src_dir / "export"
    export_dir.mkdir(exist_ok=True)
    
    export_file = export_dir / "AanExport.scala"
    export_code = """package com.example.export

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import com.example.config.ExportConfig

/**
 * Handles data export operations for AAN pipeline.
 */
class AanExport(spark: SparkSession) {
  
  private val config = ExportConfig.load()
  
  def export(data: DataFrame, outputPath: String): Unit = {
    println(s"Exporting data to $outputPath")
    
    // Add export timestamp
    val enrichedData = data.withColumn("export_timestamp", current_timestamp())
    
    // Write to output
    enrichedData
      .coalesce(config.outputPartitions)
      .write
      .mode("overwrite")
      .parquet(outputPath)
    
    println(s"Successfully exported ${enrichedData.count()} records")
  }
  
  def exportAsJson(data: DataFrame, outputPath: String): Unit = {
    data.write
      .mode("overwrite")
      .json(outputPath)
  }
}
"""
    export_file.write_text(export_code)
    
    # File 3: Notification module
    notification_dir = src_dir / "notification"
    notification_dir.mkdir(exist_ok=True)
    
    notification_file = notification_dir / "AanNotification.scala"
    notification_code = """package com.example.notification

import java.time.LocalDateTime
import scala.collection.mutable.ListBuffer

/**
 * Manages notifications for the AAN pipeline.
 */
class AanNotification {
  
  private val notifications = ListBuffer[Notification]()
  
  def sendCompletion(message: String): Unit = {
    val notification = Notification(
      message = message,
      timestamp = LocalDateTime.now(),
      level = "INFO"
    )
    
    notifications += notification
    println(s"[${notification.level}] ${notification.timestamp}: ${notification.message}")
  }
  
  def sendError(message: String, error: Throwable): Unit = {
    val notification = Notification(
      message = s"$message: ${error.getMessage}",
      timestamp = LocalDateTime.now(),
      level = "ERROR"
    )
    
    notifications += notification
    System.err.println(s"[${notification.level}] ${notification.timestamp}: ${notification.message}")
  }
  
  def getNotificationHistory(): List[Notification] = {
    notifications.toList
  }
}

case class Notification(
  message: String,
  timestamp: LocalDateTime,
  level: String
)
"""
    notification_file.write_text(notification_code)
    
    # File 4: Utils
    utils_dir = src_dir / "utils"
    utils_dir.mkdir(exist_ok=True)
    
    validator_file = utils_dir / "DataValidator.scala"
    validator_code = """package com.example.utils

import java.nio.file.{Files, Paths}

/**
 * Utility class for data validation.
 */
class DataValidator {
  
  def validatePath(path: String): Boolean = {
    if (path == null || path.isEmpty) {
      return false
    }
    
    try {
      Files.exists(Paths.get(path))
    } catch {
      case _: Exception => false
    }
  }
  
  def validateData(data: Any): Boolean = {
    data != null
  }
}
"""
    validator_file.write_text(validator_code)
    
    # File 5: Config
    config_dir = src_dir / "config"
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "ExportConfig.scala"
    config_code = """package com.example.config

/**
 * Configuration for export operations.
 */
case class ExportConfig(
  outputPartitions: Int = 10,
  compressionCodec: String = "snappy",
  maxRecordsPerFile: Long = 1000000
)

object ExportConfig {
  def load(): ExportConfig = {
    // In real implementation, would load from config file
    ExportConfig()
  }
}
"""
    config_file.write_text(config_code)
    
    print(f"Created test repository at: {test_repo}")
    return test_repo


def generate_complete_artifacts(repository_path: Path, enable_llm: bool = True):
    """Generate complete artifacts for a repository."""
    
    print(f"\n{'='*70}")
    print("GENERATING COMPLETE ARTIFACTS")
    print(f"{'='*70}")
    print(f"Repository: {repository_path}")
    print(f"LLM Summaries: {'Enabled' if enable_llm else 'Disabled'}")
    
    # Initialize components
    print("\nInitializing components...")
    classifier = FileClassifier()
    builder = IdentityCardBuilder(enable_llm_summaries=enable_llm)
    extractor = ScalaModelExtractor()
    
    # Find all Scala files
    scala_files = list(repository_path.glob("**/*.scala"))
    print(f"\nFound {len(scala_files)} Scala files:")
    for f in scala_files:
        print(f"  - {f.relative_to(repository_path)}")
    
    # Create artifacts directory structure
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = artifacts_dir / f"complete_run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    # Directories for different artifact types
    classifications_dir = run_dir / "classifications"
    classifications_dir.mkdir(exist_ok=True)
    
    entities_dir = run_dir / "entities"
    entities_dir.mkdir(exist_ok=True)
    
    relationships_dir = run_dir / "relationships"
    relationships_dir.mkdir(exist_ok=True)
    
    identity_cards_dir = run_dir / "identity_cards"
    identity_cards_dir.mkdir(exist_ok=True)
    
    # Step 1: Classify all files
    print(f"\n{'='*60}")
    print("STEP 1: FILE CLASSIFICATION")
    print(f"{'='*60}")
    
    classifications = []
    for file_path in scala_files:
        print(f"\nClassifying: {file_path.name}")
        classification = classifier.classify_file(file_path)
        classifications.append(classification)
        
        # Save classification
        class_file = classifications_dir / f"{file_path.stem}_classification.json"
        class_data = {
            "file_path": str(file_path.relative_to(repository_path)),
            "file_type": classification.classification_result.file_type.value,
            "complexity": classification.complexity_level.value,
            "frameworks": [f.value for f in classification.classification_result.frameworks],
            "metrics": {
                "lines": classification.metrics.line_count,
                "size_bytes": classification.metrics.size_bytes
            }
        }
        with open(class_file, 'w') as f:
            json.dump(class_data, f, indent=2)
        
        print(f"  Type: {classification.classification_result.file_type.value}")
        print(f"  Complexity: {classification.complexity_level.value}")
        print(f"  Lines: {classification.metrics.line_count}")
    
    # Step 2: Extract entities and relationships
    print(f"\n{'='*60}")
    print("STEP 2: ENTITY & RELATIONSHIP EXTRACTION")
    print(f"{'='*60}")
    
    file_entities = []
    for file_path in scala_files:
        print(f"\nExtracting from: {file_path.name}")
        
        try:
            result = extractor.extract_from_file(file_path)
            if result and result.file_entity:
                file_entities.append(result.file_entity)
                
                # Save entities
                entities_file = entities_dir / f"{file_path.stem}_entities.json"
                entities_data = {
                    "file_path": str(file_path.relative_to(repository_path)),
                    "imports": result.file_entity.imports,
                    "entities": [
                        {
                            "name": e.name,
                            "type": e.entity_type.value,
                            "qualified_name": e.qualified_name,
                            "line": e.location.line_start if e.location else None
                        }
                        for e in result.file_entity.entities
                    ]
                }
                with open(entities_file, 'w') as f:
                    json.dump(entities_data, f, indent=2)
                
                print(f"  Extracted {len(result.file_entity.entities)} entities")
                
                # Save relationships (if available in result)
                # Note: relationships might be in file_entity
                relationships = []
                if hasattr(result.file_entity, 'relationships'):
                    relationships = result.file_entity.relationships
                elif hasattr(result, 'relationships'):
                    relationships = result.relationships
                
                if relationships:
                    rel_file = relationships_dir / f"{file_path.stem}_relationships.json"
                    rel_data = {
                        "file_path": str(file_path.relative_to(repository_path)),
                        "relationships": [
                            {
                                "source": r.source_entity if hasattr(r, 'source_entity') else str(r),
                                "target": r.target_entity if hasattr(r, 'target_entity') else "",
                                "type": r.relationship_type.value if hasattr(r, 'relationship_type') else "unknown",
                                "line": r.location.line_start if hasattr(r, 'location') and r.location else None
                            }
                            for r in relationships
                        ]
                    }
                    with open(rel_file, 'w') as f:
                        json.dump(rel_data, f, indent=2)
                    
                    print(f"  Found {len(relationships)} relationships")
                    
        except Exception as e:
            print(f"  Error: {e}")
    
    # Step 3: Build repository graph
    print(f"\n{'='*60}")
    print("STEP 3: REPOSITORY RELATIONSHIP GRAPH")
    print(f"{'='*60}")
    
    if file_entities:
        repo_graph = extractor.analyze_repository_relationships(file_entities)
        
        # Save repository graph
        graph_file = run_dir / "repository_graph.json"
        graph_data = {
            "total_files": len(repo_graph.entities_by_file),
            "total_entities": sum(len(entities) for entities in repo_graph.entities_by_file.values()),
            "file_dependencies": {
                str(source.relative_to(repository_path)): [
                    str(target.relative_to(repository_path)) for target in targets
                ]
                for source, targets in repo_graph.file_dependencies.items()
            },
            "cross_file_relationships": len(repo_graph.cross_file_relationships)
        }
        with open(graph_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"\nRepository Graph:")
        print(f"  Files analyzed: {len(repo_graph.entities_by_file)}")
        print(f"  File dependencies: {len(repo_graph.file_dependencies)}")
        print(f"  Cross-file relationships: {len(repo_graph.cross_file_relationships)}")
    
    # Step 4: Generate identity cards with LLM summaries
    print(f"\n{'='*60}")
    print("STEP 4: IDENTITY CARDS WITH LLM SUMMARIES")
    print(f"{'='*60}")
    
    cards = builder.build_cards_with_summaries(
        repository_path=repository_path,
        classifications=classifications,
        batch_size=5
    )
    
    # Save identity cards
    for file_path, card in cards.items():
        card_file = identity_cards_dir / f"{file_path.stem}_identity.json"
        with open(card_file, 'w') as f:
            json.dump(card.to_dict(), f, indent=2)
        
        print(f"\n{card.file_name}:")
        print(f"  Entities: {len(card.file_entities)}")
        print(f"  Upstream files: {len(card.upstream_files)}")
        print(f"  Downstream files: {len(card.downstream_files)}")
        if card.llm_summary:
            print(f"  LLM Summary: {card.llm_summary[:100]}...")
    
    # Step 5: Generate summary report
    print(f"\n{'='*60}")
    print("STEP 5: SUMMARY REPORT")
    print(f"{'='*60}")
    
    summary = {
        "timestamp": timestamp,
        "repository": str(repository_path),
        "statistics": {
            "total_files": len(scala_files),
            "total_classifications": len(classifications),
            "total_entities": sum(len(fe.entities) for fe in file_entities),
            "total_identity_cards": len(cards),
            "cards_with_llm_summaries": sum(1 for card in cards.values() if card.llm_summary)
        },
        "files": [
            {
                "name": f.name,
                "path": str(f.relative_to(repository_path)),
                "entities": len(cards[f].file_entities) if f in cards else 0,
                "has_llm_summary": bool(cards[f].llm_summary) if f in cards else False
            }
            for f in scala_files
        ]
    }
    
    summary_file = run_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary:")
    print(f"  Total files processed: {summary['statistics']['total_files']}")
    print(f"  Total entities extracted: {summary['statistics']['total_entities']}")
    print(f"  Identity cards generated: {summary['statistics']['total_identity_cards']}")
    print(f"  Cards with LLM summaries: {summary['statistics']['cards_with_llm_summaries']}")
    
    print(f"\n{'='*60}")
    print(f"All artifacts saved to: {run_dir}")
    print(f"{'='*60}")
    
    return run_dir


def main():
    """Main execution function."""
    
    print("\n" + "="*80)
    print("COMPLETE ARTIFACT GENERATION SYSTEM")
    print("="*80)
    
    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    enable_llm = bool(api_key)
    
    if enable_llm:
        print(f"\nAPI key found: {api_key[:20]}...")
        print("LLM summaries will be generated.")
    else:
        print("\nNo API key found. LLM summaries will be skipped.")
        print("To enable: Add ANTHROPIC_API_KEY to .env file")
    
    # Create test repository
    print("\nCreating test repository...")
    test_repo = create_test_repository()
    
    # Generate complete artifacts
    artifacts_dir = generate_complete_artifacts(test_repo, enable_llm)
    
    # Display file tree of artifacts
    print(f"\n{'='*80}")
    print("ARTIFACT STRUCTURE")
    print(f"{'='*80}")
    
    def print_tree(path, prefix="", is_last=True):
        """Print directory tree."""
        if path.is_file():
            print(f"{prefix}{'|-- ' if not is_last else '+-- '}{path.name}")
        else:
            print(f"{prefix}{'|-- ' if not is_last else '+-- '}{path.name}/")
            children = list(path.iterdir())
            for i, child in enumerate(children):
                is_last_child = i == len(children) - 1
                extension = "    " if is_last else "|   "
                print_tree(child, prefix + extension, is_last_child)
    
    print_tree(artifacts_dir)
    
    print(f"\n{'='*80}")
    print("GENERATION COMPLETE")
    print(f"{'='*80}")
    print("\nThe system has generated:")
    print("  1. File classifications with metrics")
    print("  2. Entity extractions for each file")
    print("  3. Relationship mappings within files")
    print("  4. Cross-file dependency graph")
    print("  5. Identity cards with LLM summaries")
    print("  6. Complete summary report")
    print("\nAll artifacts are properly organized and stored for easy access.")


if __name__ == "__main__":
    main()