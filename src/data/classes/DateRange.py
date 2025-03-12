from datetime import datetime, timedelta
import requests
import pytz



class DayDateRange:
    def __init__(self, start_date_str, end_date_str, printBool):
        self.start_date_str = start_date_str
        self.end_date_str = end_date_str
        self.printBool = printBool



    def get_dates_between(self) -> list:
        """
        Return a list of dates between start_date and end_date (inclusive).
        
        Parameters:
        - start_date (str): Start date in 'YYYY-MM-DD' format
        - end_date (str): End date in 'YYYY-MM-DD' format
        
        Returns:
        - list: List of date strings in 'YYYY-MM-DD' format
        """
        # Convert strings to datetime
        try:
            start = datetime.strptime(self.start_date_str, "%Y-%m-%d")
            end = datetime.strptime(self.end_date_str, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format. Use 'YYYY-MM-DD'. {e}")

        if start > end:
            raise ValueError("start_date must be before or equal to end_date")

        # Generate list of dates
        date_list = []
        current_date = start
        while current_date <= end:
            date_str = current_date.strftime("%Y-%m-%d")
            date_dateType = datetime.strptime(date_str, "%Y-%m-%d")
            if self.printBool:
                print(date_str)
            date_list.append(date_dateType)
            current_date += timedelta(days=1)

        return date_list

# # Example usage
# if __name__ == "__main__":
#     date_range = DateRange(printBool=True)
#     dates = date_range.get_dates_between("2023-01-01", "2023-01-03")
#     print("Returned list:", dates)