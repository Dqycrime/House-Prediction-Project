import streamlit as st
import numpy as np 
from joblib import load
st. set_page_config(layout="wide")
tab1, tab2 = st.tabs(["Home", "Predict"])
fforest = load('forest.pkl')
scaler = load("scaler.pkl")
scaler_y = load('scalery.pkl')
# zipdict = {'WA 98001': 0, 'WA 98002': 1, 'WA 98003': 2, 'WA 98004': 3, 'WA 98005': 4, 'WA 98006': 5, 'WA 98007': 6, 'WA 98008': 7, 'WA 98010': 8, 'WA 98011': 9, 'WA 98014': 10, 'WA 98019': 11, 'WA 98022': 12, 'WA 98023': 13, 'WA 98024': 14, 'WA 98027': 15, 'WA 98028': 16, 'WA 98029': 17, 'WA 98030': 18, 'WA 98031': 19, 'WA 98032': 20, 'WA 98033': 21, 'WA 98034': 22, 'WA 98038': 23, 'WA 98039': 24, 'WA 98040': 25, 'WA 98042': 26, 'WA 98045': 27, 'WA 98047': 28, 'WA 98050': 29, 'WA 98051': 30, 'WA 98052': 31, 'WA 98053': 32, 'WA 98055': 33, 'WA 98056': 34, 'WA 98057': 35, 'WA 98058': 36, 'WA 98059': 37, 'WA 98065': 38, 'WA 98068': 39, 'WA 98070': 40, 'WA 98072': 41, 'WA 98074': 42, 'WA 98075': 43, 'WA 98077': 44, 'WA 98092': 45, 'WA 98102': 46, 'WA 98103': 47, 'WA 98105': 48, 'WA 98106': 49, 'WA 98107': 50, 'WA 98108': 51, 'WA 98109': 52, 'WA 98112': 53, 'WA 98115': 54, 'WA 98116': 55, 'WA 98117': 56, 'WA 98118': 57, 'WA 98119': 58, 'WA 98122': 59, 'WA 98125': 60, 'WA 98126': 61, 'WA 98133': 62, 'WA 98136': 63, 'WA 98144': 64, 'WA 98146': 65, 'WA 98148': 66, 'WA 98155': 67, 'WA 98166': 68, 'WA 98168': 69, 'WA 98177': 70, 'WA 98178': 71, 'WA 98188': 72, 'WA 98198': 73, 'WA 98199': 74, 'WA 98288': 75, 'WA 98354': 76}
# citydict = {'Algona': 0, 'Auburn': 1, 'Beaux Arts Village': 2, 'Bellevue': 3, 'Black Diamond': 4, 'Bothell': 5, 'Burien': 6, 'Carnation': 7, 'Clyde Hill': 8, 'Covington': 9, 'Des Moines': 10, 'Duvall': 11, 'Enumclaw': 12, 'Fall City': 13, 'Federal Way': 14, 'Inglewood-Finn Hill': 15, 'Issaquah': 16, 'Kenmore': 17, 'Kent': 18, 'Kirkland': 19, 'Lake Forest Park': 20, 'Maple Valley': 21, 'Medina': 22, 'Mercer Island': 23, 'Milton': 24, 'Newcastle': 25, 'Normandy Park': 26, 'North Bend': 27, 'Pacific': 28, 'Preston': 29, 'Ravensdale': 30, 'Redmond': 31, 'Renton': 32, 'Sammamish': 33, 'SeaTac': 34, 'Seattle': 35, 'Shoreline': 36, 'Skykomish': 37, 'Snoqualmie': 38, 'Snoqualmie Pass': 39, 'Tukwila': 40, 'Vashon': 41, 'Woodinville': 42, 'Yarrow Point': 43}
# cl=['Algona', 'Auburn', 'Beaux Arts Village', 'Bellevue', 'Black Diamond', 'Bothell', 'Burien', 'Carnation', 'Clyde Hill', 'Covington', 'Des Moines', 'Duvall', 'Enumclaw', 'Fall City', 'Federal Way', 'Inglewood-Finn Hill', 'Issaquah', 'Kenmore', 'Kent', 'Kirkland', 'Lake Forest Park', 'Maple Valley', 'Medina', 'Mercer Island', 'Milton', 'Newcastle', 'Normandy Park', 'North Bend', 'Pacific', 'Preston', 'Ravensdale', 'Redmond', 'Renton', 'Sammamish', 'SeaTac', 'Seattle', 'Shoreline', 'Skykomish', 'Snoqualmie', 'Snoqualmie Pass', 'Tukwila', 'Vashon', 'Woodinville', 'Yarrow Point']
# zl = ['WA 98001', 'WA 98002', 'WA 98003', 'WA 98004', 'WA 98005', 'WA 98006', 'WA 98007', 'WA 98008', 'WA 98010', 'WA 98011', 'WA 98014', 'WA 98019', 'WA 98022', 'WA 98023', 'WA 98024', 'WA 98027', 'WA 98028', 'WA 98029', 'WA 98030', 'WA 98031', 'WA 98032', 'WA 98033', 'WA 98034', 'WA 98038', 'WA 98039', 'WA 98040', 'WA 98042', 'WA 98045', 'WA 98047', 'WA 98050', 'WA 98051', 'WA 98052', 'WA 98053', 'WA 98055', 'WA 98056', 'WA 98057', 'WA 98058', 'WA 98059', 'WA 98065', 'WA 98068', 'WA 98070', 'WA 98072', 'WA 98074', 'WA 98075', 'WA 98077', 'WA 98092', 'WA 98102', 'WA 98103', 'WA 98105', 'WA 98106', 'WA 98107', 'WA 98108', 'WA 98109', 'WA 98112', 'WA 98115', 'WA 98116', 'WA 98117', 'WA 98118', 'WA 98119', 'WA 98122', 'WA 98125', 'WA 98126', 'WA 98133', 'WA 98136', 'WA 98144', 'WA 98146', 'WA 98148', 'WA 98155', 'WA 98166', 'WA 98168', 'WA 98177', 'WA 98178', 'WA 98188', 'WA 98198', 'WA 98199', 'WA 98288', 'WA 98354']
def encode_labels(label_mapping, labels):
    # Create a reverse mapping (encoded values to original labels)
    reverse_mapping = {v: k for k, v in label_mapping.items()}

    # Encode the new labels using the reverse mapping
    encoded_labels = [label_mapping[label] for label in labels if label in label_mapping]

    return encoded_labels
st.markdown(
    """
    <style>
    img {
        border-radius: 35px 35px 25px 25px;
        padding-top: 15px;
        padding-right: 10px;
    }
    .st-emotion-cache-10trblm{
        padding-top:12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


with tab1:
    
    tab1.title(":house: House Price Prediction Model (Paris)")
    col1, col2 = st.columns([3,2])
    col1.image("https://www.loans.com.au/contentAsset/image/26c57517-4aaa-4137-8377-fb170de877ee")
  
    
    col2.header("How to use this website in 3 simple steps:")
    col2.write("1. Go to the \"Predict\" tab.")
    col2.write("2. Fill in the info in the form.")
    col2.write("3. Find and press the \"Predict\" button.")

    tab1.divider()
    tab1.subheader("About This Project")
    tab1.write(" In this age of technology, we are surrounded by computers and algorithms. Computer science has always interested me, even from a young age. Artificial Intelligence allows computers to solve problems based on the data given, such as being able to think independently, like a human. The idea of a computer being able to gain human characteristics is very interesting to me. Because of this, I made this simple AI model that can predict house prices in Paris.")


# Open the app or website.

# 

# Fill in the info in the form.

# 

# Wait for the result.

# Check the displayed prediction.

# Keep it simple, and follow these steps to make your prediction!")
   



with tab2:
   st.header("Input Your Data Here:")
with tab2.form(key='details'):
    #df[['bedrooms','bathrooms','floors','sqft_basement','sqft_living','view','sqft_above','city','statezip','sqft_lot']]=X
        
        sqftm = st.slider("Area of House (Sqm)", 0, 1000,0)
        rooms = st.text_input("Rooms :bed:",value='1')
        # bathrooms = st.text_input("Bathrooms :shower:")
        floors = st.text_input("Floors",value='1')
        # abasement = st.text_input("Basement Area (0 if none)")
       
        # aliving = st.text_input("Living Area")
        
        isnewbuilt = st.checkbox('Newly Built?',value='1')
        year = st.text_input("Year Built",value='1')
        abasement = st.slider("Basement Area (0 if none)", 0, 300, 0)
        aattic = st.slider("Attic Area", 370, 300, 0)
       
        agarage = st.slider("Area of Upper Floors", 0, 300, 0)
        pool = st.checkbox('Has pool?')
        yard = st.checkbox('Has backyard?')

        # aabove = st.text_input("Area of Upper Floors")
        
        if pool:
            pool = 1
        else:
            pool=0
            
        if yard:
            yard = 1
        else:
            yard=0

        if isnewbuilt:
            isnewbuilt = 1
        else:
            isnewbuilt=0
            
        
        construct = st.form_submit_button('Predict!')
        if construct:
            a1D = np.array([sqftm,rooms,floors,year,abasement,aattic,agarage])
            tab2.divider()
            a1D = a1D.reshape(1,-1)
            a1D = scaler.transform(a1D)
            a1D = np.append(a1D, [pool, yard, isnewbuilt])
        
            tab2.subheader('Your house should cost around: ')
            result = fforest.predict(a1D.reshape(1,-1)).reshape(-1,1)
            final = scaler_y.inverse_transform(result)
            final_number = format(int(final),',')

            tab2.write(f'€{final_number}')
            
            
    
