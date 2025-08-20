name := "elt-aan-aan"

version := "1.0.0"

scalaVersion := "2.12.17"

val sparkVersion = "3.2.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "com.typesafe" % "config" % "1.4.2",
  "org.scalatest" %% "scalatest" % "3.2.12" % Test
)

// Assembly settings
assembly / assemblyMergeStrategy := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}

// Test settings
Test / parallelExecution := false