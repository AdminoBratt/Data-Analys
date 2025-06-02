import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('diamonds.csv')

st.title('Vad avgör priset på diamanter?')
st.write('Priset på diamanter beror på flera olika faktorer. Här kan vi se hur olika faktorer påverkar priset på diamanter.')
st.write('Vi börjar med att kolla på vilka priser som är de vanligaste.')

df['price'].plot(kind='hist', bins=50, figsize=(10, 6), 
                                           title='Histogram för Pris', 
                                           xlabel='Pris ($)', 
                                           ylabel='Frekvens')
fig = plt.gcf()
st.pyplot(fig=fig)
st.write('som vi ser så finns det färre av de dyrare diamanterna. Ja det är basal ekonomi med utbud och efterfrågan. Men vad är det som skiljer de dyrare diamanterna från de andra utöver raritet?')

df_regression = df.copy()

cut_mapping = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Premium': 4, 'Ideal': 5}
color_mapping = {'J': 1, 'I': 2, 'H': 3, 'G': 4, 'F': 5, 'E': 6, 'D': 7} 
clarity_mapping = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4, 'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}

df_regression['cut_num'] = df_regression['cut'].map(cut_mapping)
df_regression['color_num'] = df_regression['color'].map(color_mapping)
df_regression['clarity_num'] = df_regression['clarity'].map(clarity_mapping)

X = df_regression[['carat', 'cut_num', 'color_num', 'clarity_num']].values
y = df_regression['price'].values

X_with_intercept = np.column_stack([np.ones(len(X)), X])
coefficients = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y

# Sidebar
st.sidebar.title("Välj analys")
analysis_type = st.sidebar.radio(
                 "Välj analystyp:",
                 ("Faktoranalys", "Prisförutsägelse")
)

if analysis_type == "Faktoranalys":
                 boxplot_option = st.sidebar.selectbox(
                                  "Välj vilken faktor du vill analysera:",
                                  ("Cut", "Color", "Clarity")
                 )

                 if boxplot_option == "Cut":
                                  df.boxplot(column='price', by='cut', figsize=(12, 6))
                                  plt.title('Prisfördelning per Cut-kvalitet')
                                  plt.suptitle('')  
                                  plt.xlabel('Cut')
                                  plt.ylabel('Pris ($)')
                                  plt.xticks(rotation=45)
                                  fig = plt.gcf()
                                  st.pyplot(fig=fig)
                                  st.write("Här ser vi hur slipningens kvalitet påverkar priset på diamanter.")

                 elif boxplot_option == "Color":
                                  df.boxplot(column='price', by='color', figsize=(12, 6))
                                  plt.title('Prisfördelning per Färg')
                                  plt.suptitle('')
                                  plt.xlabel('Color (D=bäst, J=sämst)')
                                  plt.ylabel('Pris ($)')
                                  fig = plt.gcf()
                                  st.pyplot(fig=fig)
                                  st.write("Här ser vi hur färgen påverkar priset. D är den bästa färgen och J är den sämsta.")

                 elif boxplot_option == "Clarity":
                                  df.boxplot(column='price', by='clarity', figsize=(12, 6))
                                  plt.title('Prisfördelning per Clarity')
                                  plt.suptitle('')
                                  plt.xlabel('Clarity (IF=bäst, I1=sämst)')
                                  plt.ylabel('Pris ($)')
                                  plt.xticks(rotation=45)
                                  fig = plt.gcf()
                                  st.pyplot(fig=fig)
                                  st.write("Här ser vi hur klarheten påverkar priset. IF är bäst och I1 är sämst.")

elif analysis_type == "Prisförutsägelse":
                 st.sidebar.subheader("Mata in diamantens egenskaper:")
    
                 carat_input = st.sidebar.slider("Vikt (Carat):", 
                                                min_value=0.2, 
                                                max_value=5.0, 
                                                value=1.0, 
                                                step=0.1)
    
                 cut_input = st.sidebar.selectbox("Slipning (Cut):", 
                                                options=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
                                                index=2)
    
                 color_input = st.sidebar.selectbox("Färg (Color):", 
                                                  options=['D', 'E', 'F', 'G', 'H', 'I', 'J'],
                                                  index=3)
    
                 clarity_input = st.sidebar.selectbox("Klarhet (Clarity):", 
                                                    options=['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'],
                                                    index=4)
    
                 cut_num = cut_mapping[cut_input]
                 color_num = color_mapping[color_input]
                 clarity_num = clarity_mapping[clarity_input]
    
                 input_features = np.array([1, carat_input, cut_num, color_num, clarity_num])
                 predicted_price = np.dot(input_features, coefficients)
    
                 st.sidebar.write(f"**Förutsagt pris: ${predicted_price:,.0f}**")
    
                 st.subheader("Din diamant i jämförelse med andra")
    
                 plt.figure(figsize=(12, 8))
                 plt.scatter(df_regression['carat'], df_regression['price'], alpha=0.5, label='Alla diamanter')
                 plt.scatter(carat_input, predicted_price, color='red', s=200, marker='*', 
                            label=f'Din diamant (${predicted_price:,.0f})', zorder=5)
                 plt.xlabel('Vikt (Carat)')
                 plt.ylabel('Pris ($)')
                 plt.title('Din diamant jämfört med alla andra diamanter')
                 plt.legend()
                 plt.grid(True, alpha=0.3)
                 fig = plt.gcf()
                 st.pyplot(fig=fig)
    
                 col1, col2 = st.columns(2)
    
                 with col1:
                     st.write("**Dina diamantens egenskaper:**")
                     st.write(f"- Vikt: {carat_input} carat")
                     st.write(f"- Slipning: {cut_input}")
                     st.write(f"- Färg: {color_input}")
                     st.write(f"- Klarhet: {clarity_input}")
    
                 with col2:
                     st.write("**Jämförelse med genomsnittet:**")
                     avg_price = df['price'].mean()
                     price_diff = predicted_price - avg_price
                     price_diff_pct = (price_diff / avg_price) * 100
        
                     if price_diff > 0:
                         st.write(f"- ${price_diff:,.0f} dyrare än genomsnittet")
                         st.write(f"- {price_diff_pct:.1f}% över genomsnittspriset")
                     else:
                         st.write(f"- ${abs(price_diff):,.0f} billigare än genomsnittet")
                         st.write(f"- {abs(price_diff_pct):.1f}% under genomsnittspriset")
    
                 tolerance = 0.1
                 similar_diamonds = df_regression[
                     (abs(df_regression['carat'] - carat_input) <= tolerance) &
                     (df_regression['cut'] == cut_input) &
                     (df_regression['color'] == color_input) &
                     (df_regression['clarity'] == clarity_input)
                 ]
    
                 if len(similar_diamonds) > 0:
                     st.write(f"**Liknande diamanter i datasetet ({len(similar_diamonds)} st):**")
                     st.write(f"- Genomsnittspris: ${similar_diamonds['price'].mean():,.0f}")
                     st.write(f"- Prisintervall: ${similar_diamonds['price'].min():,.0f} - ${similar_diamonds['price'].max():,.0f}")

st.title('I verkligheten är diamantens genskaper sammankopplade')
st.write('Istället för att bara säga "större diamanter kostar mer" kan vi exakt säga "varje extra carat ökar priset med X, allt annat lika". jag får helt enkelt siffror på hur mycket varje egenskap påverkar priset.')

print("Numeriska värden för cut:", df_regression['cut_num'].unique())
print("Numeriska värden för color:", df_regression['color_num'].unique())
print("Numeriska värden för clarity:", df_regression['clarity_num'].unique())

print("Regressionskoefficienter:")
print(f"Intercept (konstant): {coefficients[0]:.2f}")
print(f"Carat: {coefficients[1]:.2f}")
print(f"Cut: {coefficients[2]:.2f}")
print(f"Color: {coefficients[3]:.2f}")
print(f"Clarity: {coefficients[4]:.2f}")

y_pred = X_with_intercept @ coefficients

ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(f"R-squared: {r_squared:.4f}")

df_regression['predicted_price'] = y_pred
df_regression['residuals'] = y - y_pred

print(f"\nMedelabsolutfel: ${np.mean(np.abs(df_regression['residuals'])):.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(df_regression['price'], df_regression['predicted_price'], alpha=0.5)
plt.plot([df_regression['price'].min(), df_regression['price'].max()], 
                      [df_regression['price'].min(), df_regression['price'].max()], 'r--', lw=2)
plt.xlabel('Faktiskt Pris ($)')
plt.ylabel('Förutsagt Pris ($)')
plt.title(f'Faktiskt vs Förutsagt Pris (R² = {r_squared:.3f})')
fig = plt.gcf()
st.pyplot(fig=fig)

st.write("Här har vi en modell som förutsäger priset på en diamant baserat på dess vikt, kvalitet, färg och klarhet. Vi kan se att modellen är ganska bra, men det finns fortfarande en del av variansen som inte förklaras av modellen.")

plt.figure(figsize=(10, 6))
plt.scatter(df_regression['predicted_price'], df_regression['residuals'], alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Förutsagt Pris ($)')
plt.ylabel('Residualer ($)')
plt.title('Residualplot')
fig = plt.gcf()
st.pyplot(fig=fig)

st.write("Residualanalysen visar diamanter som är ovanligt dyra eller billiga jämfört med vad modellen förväntar sig. Detta kan avslöja dolda kvalitetsfaktorer eller dålig/felaktig prissättning.")
