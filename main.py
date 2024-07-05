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

    return mean_age, median_age, mode_age, range_age, var_age, std_age


# Calcula la tasa de supervivencia general.
def calculate_survival_rate():
    s_rate = df['survived'].mean()
    return s_rate


# Calcula la tasa de supervivencia para ambos géneros.
def calculate_survival_rate_by_gender():
    survival_rate_male = df[df['gender'] == 'male']['survived'].mean()
    survival_rate_female = df[df['gender'] == 'female']['survived'].mean()

    return survival_rate_male, survival_rate_female


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


# Crea un intervalo de confianza del 95% para la edad promedio de las
# personas en el barco.
def create_confidence_interval():
    mean_age = df['age'].mean()
    std_err_age = stats.sem(df['age'])

    conf_inter = stats.t.interval(
        0.95,
        len(df['age']) - 1,
        loc=mean_age,
        scale=std_err_age
    )

    return conf_inter


# Realiza una prueba T para saber si la edad promedio de las mujeres
# interesadas en abordar al Titanic es mayor a 56.
def t_test_avg_women_age():
    ttest_result_women = stats.ttest_1samp(
        df[df['gender'] == 'female']['age'],
        56
    )

    return ttest_result_women


# Realiza una prueba T para saber si la edad promedio de los hombres
# interesados en abordar al Titanic es mayor a 56.
def t_test_avg_men_age():
    ttest_result_men = stats.ttest_1samp(df[df['gender'] == 'male']['age'], 56)
    return ttest_result_men


# Calcula la diferencia entre las tasas de supervivencia de ambos géneros.
def t_test_survival_rate_by_gender():
    survival_rate_male = df[df['gender'] == 'male']['survived']
    survival_rate_female = df[df['gender'] == 'female']['survived']

    ttest_gender_survival = stats.ttest_ind(
        survival_rate_male,
        survival_rate_female,
        equal_var=False
    )

    return ttest_gender_survival


# Calcula la diferencia entre las tasas de supervivencia de todas las clases.
def anova_test_survival_rate_by_class():
    survival_rate_class1 = df[df['p_class'] == 1]['survived']
    survival_rate_class2 = df[df['p_class'] == 2]['survived']
    survival_rate_class3 = df[df['p_class'] == 3]['survived']

    anova_result = stats.f_oneway(
        survival_rate_class1,
        survival_rate_class2,
        survival_rate_class3
    )

    return anova_result


# Realiza una prueba T para comprobar si en promedio las mujeres tenían menos
# edad que los hombres.
def t_test_age_difference_by_gender():
    ttest_age_diff = stats.ttest_ind(
        df[df['gender'] == 'female']['age'],
        df[df['gender'] == 'male']['age'],
        equal_var=False
    )

    return ttest_age_diff


# Funcion para formatear un TResult de modo que sea más legible.
def t_result_formatter(t_result):
    formatted_result = (
        f"\nEstadístico T: {t_result.statistic:.4f}\n"
        f"Valor p: {t_result.pvalue:.4e}\n"
        f"Grados de libertad: {t_result.df}\n"
    )
    return formatted_result


def f_oneway_result_formatter(f_oneway_result):
    formatted_result = (
        f"\nEstadístico F: {f_oneway_result[0]:.4f}\n"
        f"Valor p: {f_oneway_result[1]:.4e}\n"
    )
    return formatted_result


def main():
    populate_age_column()

    # Llamada a las funciones para obtener resultados
    (mean_age,
     median_age,
     mode_age,
     range_age,
     var_age,
     std_age) = calculate_age_statistics()

    print(
        f"Estadísticas en base a la edad: \n"
        f"Media: {mean_age}\n"
        f"Mediana: {median_age}\n"
        f"Moda: {mode_age}\n"
        f"Rango: {range_age}\n"
        f"Varianza: {var_age}\n"
        f"Desviación estándar: {std_age}\n"
    )

    general_survival_rate = calculate_survival_rate()
    print(f"Tasa de supervivencia general: {general_survival_rate}")

    survival_rate_by_gender = calculate_survival_rate_by_gender()
    print(f"Tasa de supervivencia de hombres: {survival_rate_by_gender[0]}")
    print(f"Tasa de supervivencia de mujeres: {survival_rate_by_gender[1]}")

    make_age_histogram_by_class()
    make_box_plot()

    confidence_interval = create_confidence_interval()
    print(
        f"\nIntervalo de confianza del 95% para la edad promedio: "
        f"{confidence_interval[0]:.5f}, {confidence_interval[1]:.5f}"
    )

    ttest_women = t_test_avg_women_age()
    print(f"\nResultado de la prueba T para mujeres: "
          f"{t_result_formatter(ttest_women)}")

    ttest_men = t_test_avg_men_age()
    print(f"Resultado de la prueba T para hombres: "
          f"{t_result_formatter(ttest_men)}")

    ttest_gender_survival = t_test_survival_rate_by_gender()
    print(
        f"Resultado de la prueba T para tasas de supervivencia por género: "
        f"{t_result_formatter(ttest_gender_survival)}"
    )

    anova_class_survival = anova_test_survival_rate_by_class()
    print(
        f"Resultado de la prueba ANOVA para tasas de supervivencia por clase: "
        f"{f_oneway_result_formatter(anova_class_survival)}"
    )

    ttest_age_diff = t_test_age_difference_by_gender()
    print(
        f"Resultado de la prueba T para diferencia de edad por género: "
        f"{t_result_formatter(ttest_age_diff)}"
    )


if __name__ == "__main__":
    main()
