class RestMexMetrics:
    """
    Clase para calcular las métricas utilizadas en la evaluación del análisis de sentimientos.
    """

    @staticmethod
    def SentimentScore(F_Sent):
        """
        Calcula el promedio de F1-scores para la clasificación de polaridad.
        
        Parámetros:
        - F_Sent: Diccionario con los valores F1 para cada clase en la tarea de clasificación de polaridad.
        
        Retorna:
        - Resp(k): Promedio de F1-scores.
        """
        C = len(F_Sent)
        return sum(F_Sent.values()) / C if C > 0 else 0

    @staticmethod
    def TypeScore(F_type):
        """
        Calcula el Macro-F1 para la clasificación de tipo.
        
        Parámetros:
        - F_type: Diccionario con los F1-scores para las clases en la tarea de clasificación de tipo.
        
        Retorna:
        - ResT(k): Macro-F1.
        """
        return sum(F_type.values()) / len(F_type) if len(F_type) > 0 else 0

    @staticmethod
    def MagicTownScore(F_town):
        """
        Calcula el promedio de F1-scores para la identificación de Pueblos Mágicos.
        
        Parámetros:
        - F_town: Diccionario con los valores F1 para cada Pueblo Mágico identificado.
        
        Retorna:
        - ResMT(k): Promedio de F1-scores.
        """
        MTL = len(F_town)
        if MTL == 0:
            raise ValueError("El diccionario F_town esta vacío.")
        return sum(F_town.values()) / MTL

    @staticmethod
    def RestMexScore(ResP_k, ResT_k, ResMT_k):
        """
        Calcula el puntaje final con pesos diferentes.
        
        Parámetros:
        - ResP_k: Promedio de F1-scores para la clasificación de polaridad.
        - ResT_k: Macro-F1 para la clasificación de tipo.
        - ResMT_k: Promedio de F1-scores para la identificación de Pueblos Mágicos.
        
        Retorna:
        - Sentiment(k): Puntaje final.
        """
        return (2 * ResP_k + ResT_k + 3 * ResMT_k) / 6

# Ejemplo de uso
if __name__ == "__main__":
    F_Sent_example = {1: 0.8, 2: 0.75, 3: 0.7, 4: 0.85, 5: 0.9}
    F_type_example = {"Attractive": 0.78, "Hotel": 0.82, "Restaurant": 0.74}
    F_town_example = {"Pueblo1": 0.8, "Pueblo2": 0.85, "Pueblo3": 0.9}

    # Calcular métricas
    ResP_k = RestMexMetrics.SentimentScore(F_Sent_example)
    ResT_k = RestMexMetrics.TypeScore(F_type_example)
    ResMT_k = RestMexMetrics.MagicTownScore(F_town_example)
    Sentiment_k = RestMexMetrics.RestMexScore(ResP_k, ResT_k, ResMT_k)

    # Mostrar resultados
    print(f"Resp(k): {ResP_k:.4f}")
    print(f"ResT(k): {ResT_k:.4f}")
    print(f"ResMT(k): {ResMT_k:.4f}")
    print(f"Sentiment(k): {Sentiment_k:.4f}")