import pandas as pd
import os

class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.datasets = {}
        
    def load_all(self):
        """Loads all Olist datasets into memory (Pandas DataFrames)."""
        print("Loading Olist datasets into memory...")
        
        files = {
            'customers': 'olist_customers_dataset.csv',
            'geolocation': 'olist_geolocation_dataset.csv',
            'order_items': 'olist_order_items_dataset.csv',
            'order_payments': 'olist_order_payments_dataset.csv',
            'order_reviews': 'olist_order_reviews_dataset.csv',
            'orders': 'olist_orders_dataset.csv',
            'products': 'olist_products_dataset.csv',
            'sellers': 'olist_sellers_dataset.csv',
            'category_translation': 'product_category_name_translation.csv'
        }
        
        for name, filename in files.items():
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                self.datasets[name] = pd.read_csv(filepath)
                print(f"Loaded {name} ({len(self.datasets[name])} rows)")
            else:
                print(f"Warning: File {filename} not found at {filepath}")
                
        # Handle datetime columns for orders and order_items
        if 'orders' in self.datasets:
            datetime_cols = [
                'order_purchase_timestamp', 
                'order_approved_at', 
                'order_delivered_carrier_date', 
                'order_delivered_customer_date', 
                'order_estimated_delivery_date'
            ]
            for col in datetime_cols:
                self.datasets['orders'][col] = pd.to_datetime(self.datasets['orders'][col], errors='coerce')
                
        if 'order_items' in self.datasets:
            self.datasets['order_items']['shipping_limit_date'] = pd.to_datetime(self.datasets['order_items']['shipping_limit_date'], errors='coerce')
            
        print("Data loading complete.")
        return self.datasets

    def get_order_details(self, order_id: str):
        """Helper to get a denormalized view of an order."""
        if not all(k in self.datasets for k in ['orders', 'order_items', 'order_payments', 'products', 'sellers', 'customers']):
            raise ValueError("All necessary datasets are not loaded.")
            
        orders = self.datasets['orders']
        order_info = orders[orders['order_id'] == order_id]
        if order_info.empty:
            return None
            
        return {
            'order': order_info.iloc[0].to_dict(),
            'items': self.datasets['order_items'][self.datasets['order_items']['order_id'] == order_id].to_dict('records'),
            'payments': self.datasets['order_payments'][self.datasets['order_payments']['order_id'] == order_id].to_dict('records')
        }

if __name__ == "__main__":
    # Test loading
    loader = DataLoader("data")
    datasets = loader.load_all()
    
    # Test getting a specific order (we'll need one from the input JSONs, but this is just to verify)
    if 'orders' in datasets and not datasets['orders'].empty:
        sample_order_id = datasets['orders'].iloc[0]['order_id']
        details = loader.get_order_details(sample_order_id)
        print(f"\\nSample details for order {sample_order_id}:")
        print(f"Order Status: {details['order']['order_status']}")
        print(f"Number of items: {len(details['items'])}")
        print(f"Number of payments: {len(details['payments'])}")
