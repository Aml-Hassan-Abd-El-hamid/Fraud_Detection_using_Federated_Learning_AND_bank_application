package com.Fintech.OnlineBanking.repository;

import java.util.Collection;


import java.util.List;
import java.util.Optional;
import java.util.Set;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import com.Fintech.OnlineBanking.domain.UserCard;
import com.Fintech.OnlineBanking.projection.CardNumberProjection;

public interface UserCardRepository extends JpaRepository <UserCard, Long> {
        
         UserCard findByCardNumber(long cardNumber);
	
	Set<UserCard> findByUserInfoNationalId(Long nationalId);
	
	


}
