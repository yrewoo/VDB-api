from src.util.logger import logger

def get_existing_solution_ids(collection, check_field, is_int=False):
    """ 이미 삽입된 check_field 목록을 가져오기 """
    try:
        existing_solutions = collection.query(
            expr=f"{check_field} >= 0" if is_int else f"{check_field} != ''",  # 모든 데이터 조회
            output_fields=[check_field]
        )
        return set(entry[check_field] for entry in existing_solutions)  # set으로 저장하여 검색 최적화
    except Exception as e:
        logger.error(e)
        return set()