import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Se crea un dataframe utilizando el archivo CSV provisto.
df = pd.read_csv('titanik.csv')


# Completa la columna 'age' con los datos correspondientes.
def populate_age_column():
    mean_age_male = df[df['gender'] == 'male']['age'].mean()
    mean_age_female = df[df['gender'] == 'female']['age'].mean()

    df.loc[
        (df['age'].isnull()) & (df['gender'] == 'male'),
        'age'
    ] = mean_age_male

    df.loc[
        (df['age'].isnull()) & (df['gender'] == 'female')
        , 'age'
    ] = mean_age_female


# Calcula las estadísticas requeridas utilizando los valores de la columna
# 'age'.
def calculate_age_statistics():
    mean_age = df['age'].mean()
    median_age = df['age'].median()
    mode_age = df['age'].mode()[0]
    range_age = df['age'].max() - df['age'].min()
    var_age = df['age'].var()
    std_age = df['age'].std()

    print(
        f"Estadísticas en base a la edad: \n"
        f"Media: {mean_age}\n"
        f"Mediana: {median_age}\n"
        f"Moda: {mode_age}\n"
        f"Rango: {range_age}\n"
        f"Varianza: {var_age}\n"
        f"Desviación estándar: {std_age}\n"
    )


# Calcula la tasa de supervivencia general.
def calculate_survival_rate():
    s_rate = df['survived'].mean()
    print(f"Tasa de supervivencia general: {s_rate}")


# Calcula la tasa de supervivencia para ambos géneros.
def calculate_survival_rate_by_gender():
    men_survival_rate = df[df['gender'] == 'male']['survived'].mean()
    women_survival_rate = df[df['gender'] == 'female']['survived'].mean()
    print(f"Tasa de supervivencia de hombres: {men_survival_rate}")
    print(f"Tasa de supervivencia de mujeres: {women_survival_rate}")


# Genera un histograma con las edades de los pasajeros por clase.
def make_age_histogram_by_class():
    classes = df['p_class'].unique()
    for p_class in classes:
        plt.figure(figsize=(10, 6))

        df[df['p_class'] == p_class]['age'].hist(
            alpha=0.5,
            label=f'Clase {p_class}'
        )

        plt.legend()
        plt.xlabel('Edad')
        plt.ylabel('Frecuencia')
        plt.title(f'Histograma de Edades para Clase {p_class}')
        plt.show()


# Genera el diagrama de cajas con las edades de los supervivientes y no
# supervivientes.
def make_box_plot():
    plt.figure(figsize=(10, 6))
    df.boxplot(column='age', by='survived')
    plt.xlabel('Supervivencia')
    plt.ylabel('Edad')
    plt.title('Diagrama de Cajas de Edades por Supervivencia')
    plt.suptitle('')
    plt.show()


# Propuesta de modelo de distribución para la edad ajustando a una de tipo
# normal.
def create_distribution_model():
    mu, sigma = stats.norm.fit(df['age'])
    print(f"\nParámetros estimados para una distribución normal: "
          f"mu = {mu}, "
          f"sigma = {sigma}"
          )


# Crea un intervalo de confianza del 95% para la edad promedio de las personas
# en el barco.
def create_confidence_interval():
    mean_age = df['age'].mean()
    std_err_age = stats.sem(df['age'])

    conf_inter = stats.t.interval(
        0.95,
        len(df['age']) - 1,
        loc=mean_age,
        scale=std_err_age
    )

    print(f"\nIntervalo de confianza del 95% para la edad promedio: "
          f"{conf_inter[0]:5f}, {conf_inter[1]:5f}")


# Realiza una prueba de hipótesis para mujeres con el objetivo de saber si el
# promedio de edad es mayor a 56.
def t_test_avg_women_age():
    women_age = df[df['gender'] == 'female']['age']
    t_stat, p_value = stats.ttest_1samp(women_age, 56)

    print(
        f"\nResultado de la prueba T para mujeres: "
        f"\nEstadístico T: {t_stat:.5f}\n"
        f"Valor p: {p_value:.5e}"
        )


# Realiza una prueba de hipótesis para hombres con el objetivo de saber si el
# promedio de edad es mayor a 56.
def t_test_avg_men_age():
    men_age = df[df['gender'] == 'male']['age']
    t_stat, p_value = stats.ttest_1samp(men_age, 56)

    print(
        f"\nResultado de la prueba T para hombres: "
        f"\nEstadístico T: {t_stat:.5f}\n"
        f"Valor p: {p_value:.5e}"
        )


# Calcula la diferencia entre las tasas de supervivencia de ambos géneros con
# un intervalo de confianza del 99%.
def survival_rate_difference():
    survival_rates = df.groupby('gender')['survived'].mean()

    n_male = len(df[df['gender'] == 'male'])
    n_female = len(df[df['gender'] == 'female'])

    # Calcula los errores estandar.
    se_male = np.sqrt(
        survival_rates['male'] * (1 - survival_rates['male']) / n_male
    )

    se_female = np.sqrt(
        survival_rates['female'] * (1 - survival_rates['female']) / n_female)

    # Calcula el intervalo de confianza para la diferencia de las tasas de
    # supervivencia.
    diff = survival_rates['female'] - survival_rates['male']

    margin_of_error = (
            stats.norm.ppf((1 + 0.99) / 2) *
            np.sqrt(se_male ** 2 + se_female ** 2)
    )

    ci = (diff - margin_of_error, diff + margin_of_error)

    print(
        f"\nResultado de la diferencia entre las tasas de supervivencia por "
        f"género con intervalo de confianza del 99%:\n"
        f"Diferencia: {diff}\n"
        f"Extremos del intervalo de confianza: {ci[0]:5f}, {ci[1]:5f}"
    )


# Calcula la diferencia entre las tasas de supervivencia de todas las clases,
# para ello se utiliza un Análisis de la Varianza (ANOVA).
def anova_test_survival_rate_by_class():
    survival_rate_class1 = df[df['p_class'] == 1]['survived']
    survival_rate_class2 = df[df['p_class'] == 2]['survived']
    survival_rate_class3 = df[df['p_class'] == 3]['survived']

    anova_result = stats.f_oneway(
        survival_rate_class1,
        survival_rate_class2,
        survival_rate_class3
    )

    print(
        f"\nDiferencia entre las tasas de supervivencia de todas las clases:"
        f"\nEstadístico F: {anova_result[0]:.5f}"
        f"\nValor p: {anova_result[1]:.5e}"
        )


# Realiza una prueba T para comprobar si en promedio las mujeres tenían menos
# edad que los hombres.
def t_test_age_difference_by_gender():
    # Primero se filtran las edades por género.
    ages_female = df[df['gender'] == 'female']['age']
    ages_male = df[df['gender'] == 'male']['age']

    # Se realiza una prueba T de dos muestras.
    t_stat, p_value = stats.ttest_ind(ages_female, ages_male)

    # Se calcula la diferencia de medias y desviación estándar de la
    # diferencia.
    mean_diff = np.mean(ages_female) - np.mean(ages_male)
    std_diff = np.sqrt((np.var(ages_female) / len(ages_female)) +
                       (np.var(ages_male) / len(ages_male)))

    # Se calcula el intervalo de confianza del 95% para la diferencia de
    # medias.
    dof = len(ages_female) + len(ages_male) - 2  # Grados de libertad
    margin_of_error = stats.t.ppf(0.975, dof) * std_diff
    ci = (mean_diff - margin_of_error, mean_diff + margin_of_error)

    # Imprimir resultados
    print(
        f"\nResultado de la prueba T para la diferencia en edad promedio "
        f"entre mujeres y hombres:"
        f"\nEstadístico T: {t_stat:.5f}"
        f"\nValor p: {p_value:.5e}"
        f"\nIntervalo de confianza del 95% para la diferencia de medias: "
        f"{ci[0]:5f}, {ci[1]:5f}"
    )


def main():
    # PARTE 1
    populate_age_column()
    calculate_age_statistics()
    calculate_survival_rate()
    calculate_survival_rate_by_gender()
    make_age_histogram_by_class()
    make_box_plot()

    # PARTE 2
    create_confidence_interval()
    create_distribution_model()
    t_test_avg_women_age()
    t_test_avg_men_age()
    survival_rate_difference()
    anova_test_survival_rate_by_class()
    t_test_age_difference_by_gender()


if __name__ == "__main__":
    main()
