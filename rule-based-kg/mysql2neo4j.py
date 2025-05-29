import mysql.connector
from mysql.connector import Error
from neo4j import GraphDatabase

NEO4J_URI = "neo4j://localhost"
NEO4J_AUTH = ("neo4j", "password")


def create_connection(host_name, user_name, user_password, db_name):
    """ 创建数据库连接 """
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            password=user_password,
            database=db_name
        )
        print("连接成功")
    except Error as e:
        print(f"连接错误: {e}")
    return connection


def execute_query(connection, query):
    """ 执行查询 """
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("查询成功")
    except Error as e:
        print(f"查询错误: {e}")


def insert_sku(tx, pri_id, spu_id, sku_name, spu_name):
    result = tx.run("""
        CREATE (n:SkuInfo {pri_id: $pri_id, spu_id: $spu_id, sku_name: $sku_name, spu_name: $spu_name}) RETURN n
        """, pri_id=pri_id, spu_id=spu_id, sku_name=sku_name, spu_name=spu_name)

    print(result)


def insert_sku_attr_value(tx, pri_id, attr_id, value_id, attr_name, value_name):
    result = tx.run("""
        CREATE (n:SkuAttrValueInfo {pri_id: $pri_id, attr_id: $attr_id, value_id: $value_id, attr_name: $attr_name, value_name: $value_name}) RETURN n
        """, pri_id=pri_id, attr_id=attr_id, value_id=value_id, attr_name=attr_name, value_name=value_name)

    print(result)


def create_relation(tx, sku_id, attr_id):
    result = tx.run("""
        MATCH (m:SkuInfo {pri_id: $sku_id}), (n:SkuAttrValueInfo {attr_id: $attr_id})
        CREATE (m)-[:HAVE]->(n)""", sku_id=sku_id, attr_id=attr_id)


def main():
    # 数据库连接信息
    host = "localhost"
    user = "root"
    password = "root"
    database = "gmall"

    # 创建连接
    connection = create_connection(host, user, password, database)

    # 查询数据
    select_query = "SELECT * FROM sku_info;"
    cursor = connection.cursor()
    cursor.execute(select_query)
    rows = cursor.fetchall()

    # 查询spu数据
    select_query_from_spu_info = "SELECT * FROM spu_info;"
    cursor = connection.cursor()
    cursor.execute(select_query_from_spu_info)
    spu_info_rows = cursor.fetchall()

    # 查询attr value的数据
    select_query_from_sku_attr_value = "select * from sku_attr_value;"
    cursor = connection.cursor()
    cursor.execute(select_query_from_sku_attr_value)
    sku_attr_value_rows = cursor.fetchall()

    with GraphDatabase.driver(uri=NEO4J_URI, auth=NEO4J_AUTH) as d:
        d.verify_connectivity()

        with d.session(database="neo4j") as session:
            for row in rows:
                pri_id = row[0]
                spu_id = row[1]
                spu_name = "unknown"
                for spu in spu_info_rows:
                    if spu[0] == spu_id:
                        spu_name = spu[1]
                price = row[2]
                sku_name = row[3]
                sku_desc = row[4]
                session.execute_write(insert_sku, pri_id,
                                      spu_id, sku_name, spu_name)

            attr_value_set = set()
            for row in sku_attr_value_rows:
                pri_id = row[0]
                attr_id = row[1]
                value_id = row[2]
                sku_id = row[3]
                attr_name = row[4]
                value_name = row[5]
                if (attr_id, value_id) not in attr_value_set:
                    session.execute_write(
                        insert_sku_attr_value, pri_id, attr_id, value_id, attr_name, value_name)
                session.execute_write(create_relation, sku_id, attr_id)
                attr_value_set.add((attr_id, value_id))

    # 关闭连接
    if connection.is_connected():
        cursor.close()
        connection.close()
        print("数据库连接已关闭")


if __name__ == "__main__":
    main()
