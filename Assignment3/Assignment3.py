import sys

from pyspark import StorageLevel
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession


def main():
    appName = "assignment3"
    master = "local"
    spark = (
        SparkSession.builder.appName(appName)
        .master(master)
        .config(
            "spark.jars",
            "/Users/bitaetaati/PythonProjectTemplate/PythonProjectTemplate/mariadb-java-client-3.0.8.jar",
        )
        .getOrCreate()
    )

    sql1 = "select * from baseball.batter_counts"
    database = "baseball"
    user = "bita"
    server = "localhost"
    port = 3306
    jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"

    df1 = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("query", sql1)
        .option("user", user)
        .option("password")
        .option("driver", jdbc_driver)
        .load()
    )

    df1.show()
    df1.printSchema()

    sql2 = "select * from baseball.game"
    database = "baseball"
    user = "bita"
    server = "localhost"
    port = 3306
    jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"

    df2 = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("query", sql2)
        .option("user", user)
        .option("password")
        .option("driver", jdbc_driver)
        .load()
    )

    df2.show()

    df1.createOrReplaceTempView("batter_counts")
    df2.createOrReplaceTempView("game")
    df1.persist(StorageLevel.DISK_ONLY)
    df2.persist(StorageLevel.DISK_ONLY)

    results = spark.sql(
        """drop table if exists rolling_batting_average;
        create table rolling_batting_average (with t1 as
        (select btc.batter, max(gm.local_date) as max_date, btc.game_id from batter_counts btc
        left join game gm
        on btc.game_id = gm.game_id
        group by btc.batter, btc.game_id),
        t2 as
        (select btc.batter, sum(btc.hit)/sum(btc.atBat) as batting_average,
        gm.local_date, case when btc.atBat = 0 then 'zero' end as atBat
        from batter_counts btc
        left join game gm
        on btc.game_id = gm.game_id
        group by btc.batter, btc.game_id)
        select t2.batter , avg(t2.batting_average)  from t2
        right join t1 on t2.batter = t1.batter
        where t2.local_date > date_add(t1.max_date, INTERVAL -100 DAY)
        group by t1.batter, t1.game_id)"""
    )
    results.show()

    # Random Forest
    random_forest = RandomForestRegressor(
        labelCol="batter",
        featuresCol="batting_average",
        numTrees=100,
        predictionCol="pred_survived",
    )
    random_forest_fitted = random_forest.fit(results)
    titanic_df = random_forest_fitted.transform(results)
    titanic_df.show()

    return


if __name__ == "__main__":
    sys.exit(main())
