class ClinicalDecisionRules:
    @staticmethod
    def bnp_interpretation(bnp):
        if bnp is None:
            return "غير متوفر", "لم يتم قياس BNP", "unknown"
        if bnp < 100:
            return "طبيعي", "فشل قلب مستبعد جدًا", "low"
        elif bnp < 400:
            return "مرتفع متوسط", "احتمالية فشل قلب متوسطة", "moderate"
        else:
            return "مرتفع جدًا", "فشل قلب محتمل بقوة", "high"

    @staticmethod
    def heart_failure_score(orthopnea, leg_swelling, weight_gain, bnp_level, jvd=None):
        score = 0
        reasons = []
        if orthopnea == 1:
            score += 2
            reasons.append("Orthopnea")
        if leg_swelling == 1:
            score += 1
            reasons.append("Leg swelling")
        if weight_gain == 1:
            score += 1
            reasons.append("Weight gain")
        if bnp_level == 'high':
            score += 2
            reasons.append("BNP high")
        elif bnp_level == 'moderate':
            score += 1
            reasons.append("BNP moderate")
        if jvd == 1:
            score += 1
            reasons.append("JVD")
        if score >= 4:
            return "عالية", f"فشل قلب محتمل ({score} points)", reasons
        elif score >= 2:
            return "متوسطة", f"احتمالية فشل قلب ({score} points)", reasons
        else:
            return "منخفضة", "فشل قلب مستبعد", reasons

    @staticmethod
    def emergency_indicators(age, rr, spo2, bnp, troponin, chest_pain):
        red_flags = []
        if age > 80:
            red_flags.append("Age > 80")
        if rr > 30:
            red_flags.append("RR > 30")
        if spo2 and spo2 < 90:
            red_flags.append("SpO2 < 90%")
        if bnp and bnp > 1000:
            red_flags.append("BNP > 1000")
        if troponin and troponin > 0.1:
            red_flags.append("Troponin elevated")
        if chest_pain == 1 and bnp and bnp > 400:
            red_flags.append("Chest pain + high BNP")
        if len(red_flags) >= 2:
            return "CRITICAL", red_flags
        elif len(red_flags) >= 1:
            return "HIGH", red_flags
        else:
            return "LOW", red_flags
